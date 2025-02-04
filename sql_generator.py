import streamlit as st
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
import sqlite3
import psycopg2
import pyodbc
import logging
import time
import copy
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseChatApp:
    def __init__(self):
        # Initialize session state variables
        session_states = [
            'chat_history', 'db_connection', 'db_cursor', 
            'editing_index', 'db_type', 'db_config', 
            'last_query_results'
        ]
        for state in session_states:
            if state not in st.session_state:
                st.session_state[state] = None if state not in ['chat_history', 'last_query_results'] else []

    def reset_database_connection(self):
        """
        Reset database connection state
        """
        st.session_state.db_connection = None
        st.session_state.db_cursor = None
        st.session_state.db_type = None
        st.session_state.db_config = None

    def connect_database(self):
        """
        Sidebar for database connection with enhanced error handling
        """
        st.sidebar.header("Database Connection")
        
        # Database type selection with reset option
        db_type = st.sidebar.selectbox(
            "Select Database Type", 
            ["Select", "SQLite", "PostgreSQL", "MySQL", "SQL Server"],
            index=0,
            key="db_type_selector"
        )

        # Reset connection if needed
        if db_type == "Select":
            self.reset_database_connection()
            return

        # Connection forms based on database type
        connection_details = {}
        
        if db_type == "SQLite":
            db_path = st.sidebar.file_uploader(
                "Upload SQLite Database", 
                type=['db', 'sqlite']
            )
            if db_path:
                try:
                    conn = sqlite3.connect(db_path.name)
                    cursor = conn.cursor()
                    
                    # Verify connection by fetching tables
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    st.session_state.db_connection = conn
                    st.session_state.db_cursor = cursor
                    st.session_state.db_type = db_type
                    st.session_state.db_config = {"path": db_path.name}
                    
                    st.sidebar.success(f"Connected to SQLite! Found {len(tables)} tables.")
                except Exception as e:
                    st.sidebar.error(f"SQLite Connection Error: {str(e)}")
                    logger.error(f"SQLite Connection Error: {traceback.format_exc()}")
        
        elif db_type in ["PostgreSQL", "MySQL", "SQL Server"]:
            with st.sidebar.form(f"{db_type.lower()}_connection", clear_on_submit=False):
                st.write(f"{db_type} Connection Details")
                
                # Common connection fields
                host = st.text_input("Host", key=f"{db_type}_host")
                port = st.text_input("Port", 
                    value={"PostgreSQL": "5432", "MySQL": "3306", "SQL Server": "1433"}[db_type], 
                    key=f"{db_type}_port"
                )
                database = st.text_input("Database Name", key=f"{db_type}_database")
                username = st.text_input("Username", key=f"{db_type}_username")
                password = st.text_input("Password", type="password", key=f"{db_type}_password")
                
                connect_button = st.form_submit_button("Connect")
                
                if connect_button:
                    connection_details = {
                        "host": host,
                        "port": port,
                        "database": database,
                        "username": username,
                        "password": password
                    }
                    
                    try:
                        if db_type == "PostgreSQL":
                            # Updated PostgreSQL connection handling
                            conn = psycopg2.connect(
                                host=host,
                                port=port,
                                database=database,
                                user=username,
                                password=password
                            )
                            # Set autocommit after connection
                            conn.autocommit = True
                            cursor = conn.cursor()
                            
                            # Verify connection
                            cursor.execute("""
                                SELECT tablename FROM pg_tables 
                                WHERE schemaname = 'public' LIMIT 5;
                            """)
                            tables = cursor.fetchall()

                        elif db_type == "MySQL":
                            conn = mysql.connector.connect(
                                host=host,
                                port=port,
                                database=database,
                                user=username,
                                password=password
                            )
                            cursor = conn.cursor()
                            
                            # Verify connection
                            cursor.execute("""
                                SELECT table_name FROM information_schema.tables 
                                WHERE table_schema = DATABASE() LIMIT 5;
                            """)
                            tables = cursor.fetchall()

                        elif db_type == "SQL Server":
                            conn_str = (
                                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                                f"SERVER={host},{port};"
                                f"DATABASE={database};"
                                f"UID={username};"
                                f"PWD={password}"
                            )
                            conn = pyodbc.connect(conn_str)
                            cursor = conn.cursor()
                            
                            # Verify connection
                            cursor.execute("""
                                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
                                WHERE TABLE_TYPE = 'BASE TABLE' LIMIT 5;
                            """)
                            tables = cursor.fetchall()

                        # Store connection details
                        st.session_state.db_connection = conn
                        st.session_state.db_cursor = cursor
                        st.session_state.db_type = db_type
                        st.session_state.db_config = connection_details
                        
                        st.sidebar.success(f"Connected to {db_type}! Found {len(tables)} tables.")
                        
                    except psycopg2.Error as e:
                        # Specific error handling for PostgreSQL
                        error_msg = f"PostgreSQL Connection Error: {str(e)}"
                        st.sidebar.error(error_msg)
                        st.sidebar.error(f"Error Code: {e.pgcode}")
                        st.sidebar.error(f"Error Message: {e.pgerror}")
                        logger.error(f"PostgreSQL Connection Error: {traceback.format_exc()}")
                        self.reset_database_connection()
                    
                    except Exception as e:
                        st.sidebar.error(f"{db_type} Connection Error: {str(e)}")
                        logger.error(f"{db_type} Connection Error: {traceback.format_exc()}")
                        self.reset_database_connection()

    def get_table_info(self):
        """
        Retrieve table information with robust error handling
        """
        if not st.session_state.db_cursor:
            st.warning("No active database connection.")
            return []

        try:
            # Reset transaction for PostgreSQL if needed
            if st.session_state.db_type == 'PostgreSQL':
                try:
                    st.session_state.db_connection.rollback()
                except Exception as rollback_error:
                    logger.warning(f"Rollback error: {rollback_error}")

            # Retrieve tables based on database type
            if st.session_state.db_type == "SQLite":
                st.session_state.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            elif st.session_state.db_type == "PostgreSQL":
                st.session_state.db_cursor.execute("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public';
                """)
            elif st.session_state.db_type == "MySQL":
                st.session_state.db_cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = DATABASE();
                """)
            elif st.session_state.db_type == "SQL Server":
                st.session_state.db_cursor.execute("""
                    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE';
                """)
            else:
                st.warning("Unsupported database type")
                return []

            tables = [table[0] for table in st.session_state.db_cursor.fetchall()]
            return tables

        except Exception as e:
            st.error(f"Error retrieving tables: {str(e)}")
            logger.error(f"Table Retrieval Error: {traceback.format_exc()}")
            return []

    def get_schema_info(self, table):
        """
        Retrieve schema information for a specific table with error handling
        """
        if not st.session_state.db_cursor:
            return []

        try:
            if st.session_state.db_type == "SQLite":
                st.session_state.db_cursor.execute(f"PRAGMA table_info({table});")
            elif st.session_state.db_type == "PostgreSQL":
                st.session_state.db_cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}';
                """)
            elif st.session_state.db_type == "MySQL":
                st.session_state.db_cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}' 
                    AND table_schema = DATABASE();
                """)
            elif st.session_state.db_type == "SQL Server":
                st.session_state.db_cursor.execute(f"""
                    SELECT COLUMN_NAME, DATA_TYPE 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{table}';
                """)
            else:
                st.warning("Unsupported database type")
                return []

            columns = st.session_state.db_cursor.fetchall()
            return columns

        except Exception as e:
            st.error(f"Error retrieving schema for {table}: {str(e)}")
            logger.error(f"Schema Retrieval Error: {traceback.format_exc()}")
            return []

    def setup_llm(self):
        """
        Setup Ollama LLM with DuckDB-NSQL for SQL translation
        """
        try:
            llm = OllamaLLM(model="duckdb-nsql")
            return llm
        except Exception as e:
            st.error(f"LLM Setup Error: {e}")
            return None

    def generate_sql_query(self, llm, table_info, user_question):
        """
        Generate SQL query using Ollama LLM
        """
        try:
            prompt_template = f"""
            Given the database schema:
            {table_info}

            Translate the following natural language question into a SQL query:
            Question: {user_question}

            SQL Query:
            """

            # Use LLM to generate SQL query
            sql_query = llm.invoke(prompt_template)
            return sql_query.strip()
        except Exception as e:
            st.error(f"SQL Generation Error: {e}")
            return None

    def execute_query(self, sql_query):
        """
        Execute SQL query and return results as DataFrame
        """
        try:
            st.session_state.db_cursor.execute(sql_query)
            columns = [column[0] for column in st.session_state.db_cursor.description]
            results = st.session_state.db_cursor.fetchall()
            return pd.DataFrame(results, columns=columns)
        except Exception as e:
            st.error(f"Query Execution Error: {e}")
            return None

    def start_edit(self, index):
        """
        Start editing a specific chat history item
        """
        st.session_state.editing_index = index

    def regenerate_query(self, index, llm, table_schema_info, edited_question):
        """
        Regenerate SQL query and results for an edited question
        """
        try:
            # Regenerate SQL Query
            sql_query = self.generate_sql_query(llm, table_schema_info, edited_question)
            
            if sql_query:
                # Execute Query
                results = self.execute_query(sql_query)
                
                if results is not None:
                    # Update the chat history with new question and response
                    st.session_state.chat_history[index]["content"] = edited_question
                    st.session_state.chat_history[index + 1] = {
                        "role": "assistant", 
                        "content": f"SQL Query:\n```sql\n{sql_query}\n```"
                    }
                    
                    # Store last query results separately
                    st.session_state.last_query_results = results
                    
                    return True
            
            return False
        except Exception as e:
            st.error(f"Error regenerating query: {e}")
            return False

    def chat_interface(self):
        """
        Main chat interface for the app
        """
        st.title("🤖 Database Chat: Natural Language Queries")

        # Check database connection
        if not st.session_state.db_connection:
            st.warning("Please connect to a database first!")
            return

        # Get tables and prepare schema info
        tables = self.get_table_info()
        if not tables:
            st.warning("No tables found in the database.")
            return

        table_schema_info = ""
        for table in tables:
            columns = self.get_schema_info(table)
            table_schema_info += f"Table: {table}\n"
            for column in columns:
                table_schema_info += f"- {column[0]} ({column[1]})\n"
            table_schema_info += "\n"

        # Setup LLM
        llm = self.setup_llm()
        if not llm:
            return

        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if st.session_state.editing_index == i and message['role'] == 'user':
                # Edit mode for user message
                with st.chat_message("user"):
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        edited_question = st.text_input(
                            "Edit your question", 
                            value=message["content"], 
                            key=f"edit_input_{i}"
                        )
                    with col2:
                        save_edit = st.button("Save", key=f"save_edit_{i}")
                        cancel_edit = st.button("Cancel", key=f"cancel_edit_{i}")
                    
                    if save_edit and edited_question:
                        # Attempt to regenerate query with edited question
                        success = self.regenerate_query(i, llm, table_schema_info, edited_question)
                        if success:
                            # Exit edit mode
                            st.session_state.editing_index = None
                            st.rerun()
                        else:
                            st.error("Failed to regenerate query. Please try again.")
                    
                    if cancel_edit:
                        st.session_state.editing_index = None
                        st.rerun()
            else:
                # Normal display of chat history
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Add edit button for user messages right after the response
                    if (message['role'] == 'user' and 
                        i + 1 < len(st.session_state.chat_history) and 
                        st.session_state.chat_history[i + 1]['role'] == 'assistant'):
                        
                        edit_button = st.button("Edit", key=f"edit_{i}")
                        if edit_button:
                            self.start_edit(i)
                            st.rerun()

        # Display last query results (if any)
        if st.session_state.last_query_results is not None:
            st.dataframe(st.session_state.last_query_results)

        # Chat input
        if user_question := st.chat_input("Ask a question about your database"):
            # Reset last query results
            st.session_state.last_query_results = None
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_question
            })

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)

            # Generate SQL query
            with st.chat_message("assistant"):
                with st.spinner("Generating SQL query..."):
                    sql_query = self.generate_sql_query(llm, table_schema_info, user_question)
                
                if sql_query:
                    st.code(sql_query, language="sql")

                    # Execute query
                    with st.spinner("Executing query..."):
                        results = self.execute_query(sql_query)
                    
                    if results is not None:
                        # Add query result to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"SQL Query:\n```sql\n{sql_query}\n```"
                        })

                        # Store and display results
                        st.session_state.last_query_results = results
                        st.dataframe(results)
                        st.success("Query executed successfully!")

    def run(self):
        """
        Main application runner
        """
        st.set_page_config(
            page_title="Database Chat", 
            page_icon="🤖",
            layout="wide"
        )
        
        # Add a container for connection status
        status_container = st.container()
        
        # Database connection sidebar
        self.connect_database()
        
        # Display connection status
        if st.session_state.db_connection:
            with status_container:
                st.success(f"Connected to {st.session_state.db_type} Database")
        
        # Main chat interface
        self.chat_interface()

# Run the application
if __name__ == "__main__":
    app = DatabaseChatApp()
    app.run()