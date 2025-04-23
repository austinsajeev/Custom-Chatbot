from dotenv import load_dotenv
load_dotenv()  # This will automatically load the .env file

import os
import sys
import argparse
import pandas as pd
import PyPDF2
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import tkinter as tk
from tkinter import scrolledtext
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def read_pdf(self, file_path):
        """Extract text from a PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            return ""

    def read_txt(self, file_path):
        """Read text from a TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            return ""

    def read_csv(self, file_path):
        """Read text from a CSV file and convert to string representation"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return ""

    def process_file(self, file_path):
        """Process a file based on its extension"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return self.read_pdf(file_path)
        elif ext == '.txt':
            return self.read_txt(file_path)
        elif ext == '.csv':
            return self.read_csv(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""

    def load_documents(self):
        """Load all documents from the data directory"""
        all_text = ""
        
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            if os.path.isfile(file_path):
                logger.info(f"Processing file: {filename}")
                file_text = self.process_file(file_path)
                all_text += file_text + "\n\n"
        
        return all_text

    def split_text(self, text):
        """Split text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)

class ChatbotEngine:
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self.memory = None
        self.qa_chain = None
        
    def initialize(self, text_chunks):
        """Initialize the chatbot engine with text chunks"""
        logger.info("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings()
        
        logger.info("Creating vector store...")
        self.vectorstore = FAISS.from_texts(text_chunks, self.embeddings)
        
        logger.info("Setting up conversation memory...")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        logger.info(f"Initializing language model: {self.model_name}")
        llm = HuggingFaceHub(
            repo_id=self.model_name,
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )
        
        logger.info("Creating conversational chain...")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )
        
        logger.info("Chatbot engine initialization complete")
    
    def get_response(self, query):
        """Get response from the chatbot for a given query"""
        if not self.qa_chain:
            return "Chatbot is not initialized yet. Please wait."
        
        try:
            response = self.qa_chain({"question": query})
            return response["answer"]
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

class ChatbotGUI:
    def __init__(self, chatbot_engine):
        self.chatbot_engine = chatbot_engine
        self.is_initialized = False
        self.init_thread = None
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Custom Data Chatbot")
        self.root.geometry("800x600")
        
        # Create the chat display area
        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Create the input area
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.input_field = tk.Entry(self.input_frame)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self.send_message)
        
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # Add status bar
        self.status_bar = tk.Label(self.root, text="Initializing chatbot...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Welcome message
        self.display_message("System", "Welcome to Custom Data Chatbot! The system is initializing...")
        
    def initialize_chatbot(self, text_chunks):
        """Initialize the chatbot in a separate thread"""
        try:
            self.chatbot_engine.initialize(text_chunks)
            self.root.after(0, self.on_init_complete)
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.root.after(0, lambda: self.update_status(f"Error initializing: {str(e)}"))
    
    def on_init_complete(self):
        """Called when initialization is complete"""
        self.is_initialized = True
        self.update_status("Chatbot ready. You can start chatting!")
        self.display_message("System", "Initialization complete. How can I help you today?")
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_bar.config(text=message)
    
    def display_message(self, sender, message):
        """Display a message in the chat window"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self, event=None):
        """Send a message and get a response"""
        user_message = self.input_field.get().strip()
        if not user_message:
            return
        
        self.input_field.delete(0, tk.END)
        self.display_message("You", user_message)
        
        if not self.is_initialized:
            self.display_message("System", "Please wait, the chatbot is still initializing...")
            return
        
        # Get response in a separate thread to keep UI responsive
        threading.Thread(target=self.get_response, args=(user_message,), daemon=True).start()
    
    def get_response(self, user_message):
        """Get a response from the chatbot"""
        self.update_status("Thinking...")
        response = self.chatbot_engine.get_response(user_message)
        
        # Update UI in the main thread
        self.root.after(0, lambda: self.display_bot_response(response))
    
    def display_bot_response(self, response):
        """Display the bot's response"""
        self.display_message("Bot", response)
        self.update_status("Ready")
    
    def start(self, text_chunks):
        """Start the chatbot GUI"""
        # Start initialization in a separate thread
        self.init_thread = threading.Thread(target=self.initialize_chatbot, args=(text_chunks,), daemon=True)
        self.init_thread.start()
        
        # Start the main loop
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Custom Data Chatbot")
    parser.add_argument("--data_dir", default="./data", help="Directory containing data files")
    parser.add_argument("--model", default="google/flan-t5-base", help="HuggingFace model to use")
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting document processing from {args.data_dir}")
        processor = DocumentProcessor(data_dir=args.data_dir)
        documents_text = processor.load_documents()
        
        if not documents_text.strip():
            logger.error("No text content found in the provided documents.")
            print("Error: No text content was found in the provided documents.")
            return
        
        text_chunks = processor.split_text(documents_text)
        logger.info(f"Created {len(text_chunks)} text chunks")
        
        chatbot_engine = ChatbotEngine(model_name=args.model)
        gui = ChatbotGUI(chatbot_engine)
        gui.start(text_chunks)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()