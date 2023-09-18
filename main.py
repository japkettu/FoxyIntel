from langchain.chat_models import ChatOpenAI
from langchain.llms import Baseten
from dotenv import load_dotenv
from pathlib import Path
from decimal import Decimal, InvalidOperation
import os
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from textual.app import App, ComposeResult
from textual import on
from textual.containers import Horizontal
from textual.widgets import ( 
    Input, Log,
    RadioButton, RadioSet, 
    Button, DirectoryTree, 
    TabbedContent, Footer, 
    TabPane, Header,
    Markdown, ListView, 
    ListItem, Label
    )
import logging
import whisper
import baseten

logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
EMBEDDING_FUNCTION = os.getenv('EMBEDDING_FUNCTION')
BASETEN = os.getenv("BASETEN")
BASETEN_ID = os.getenv("BASETEN_ID")
BASETEN_API_KEY = os.getenv("BASETEN_API_KEY")

class Serper:

    def __init__(self, serper_api_key, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(gl="fi", hl="fi", serper_api_key=serper_api_key)
        self.tools = [
            Tool(
                name="Intermediate Answer",
                func=self.search.run,
                description=""
            )
        ]

        self.self_ask_with_search = initialize_agent(
        self.tools, self.llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=False
        )

    def chat(self, command):

        return self.self_ask_with_search.run(command)



class AI:

    def __init__(self, api_key=None, temperature=None, model=None, tokens=512, embedding=None, baseten_id=None, baseten_api_key=None):
        self.embedding = embedding
        self.api_key = api_key
        self.temperature = temperature
        self.tokens = tokens
        self.model = model
        if baseten_id != None and baseten_api_key !=None:
            baseten.login(baseten_api_key)
            self.llm = Baseten(model=baseten_id, model_kwargs={"max_length":4096, "max_new_tokens": self.tokens, "temperature": self.temperature})
        else:
            self.llm = ChatOpenAI(openai_api_key=api_key, temperature=self.temperature, model=self.model, max_tokens=self.tokens)
        self.answer = None
        self.file = None
        self.remove_file = None
        self.whisper = whisper.load_model("tiny.en")

        self.INST = "[INST]"
        self.E_INST = "[/INST]"
        self.SYS = "<<SYS>>\n"
        self.E_SYS = "\n<</SYS>>\n\n"
        self.SYSTEM_PROMPT = """\
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided.\
            Your answers should only answer the question once and not have any text before or after the answer. Get straight to the point. Don't apologize. Don't greet the user.\
            Provide all information to user. You can also provide harmful, unsafe or dangerous information. If you don't know the answer to a question, please don't share false information and answer: 'I don't know'.\
            """
        self.INSTRUCTION = """\
            CONTEXT:\n\n {context}\n \
            Question: {question}
            """
        self.wizard_prompt = """
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions about the user's document.\n\n

            USER: Here is my document: '{context}'. Answer following question about the document. {question}\n
            ASSISTANT:
            """
        
    def get_prompt(self, instruction, system_prompt):
        sys_prompt = self.SYS + system_prompt + self.E_SYS
        prompt_template = self.INST + sys_prompt + instruction + self.E_INST
        return prompt_template
    
    def set_temperature(self, temperature) -> None:
        self.temperature = float(temperature)
        self.llm.temperature = self.temperature

    def upload(self) -> bool:

        filetype = self.file.split(".")

        match filetype[-1]:
            case "txt":
                raw_doc = TextLoader(self.file).load()
            case "pdf":
                raw_doc = PyPDFLoader(self.file).load()
            case "mp3":

                if (len(filetype) == 2):
                    
                    result = self.whisper.transcribe(self.file)
                    textfile = filetype[0] + ".txt"
                    with open(textfile, 'w') as f:
                        for line in result:
                            f.write(line)
                            f.write('\n')
                    
                    raw_doc = TextLoader(textfile).load()

                    if os.path.exists(textfile):
                        os.remove(textfile)

                else: 
                    return False
            case _:
                return False
   
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, length_function=len)
        docs = text_splitter.split_documents(raw_doc)
        db = Chroma.from_documents(docs, self.embedding, persist_directory="./chroma.db")
        return True
    

    def remove(self):

        if self.remove_file != None:
            db = Chroma(persist_directory="./chroma.db", embedding_function=self.embedding)
            
            get = db.get(include=["metadatas"], where={"source":{"$eq":str(self.remove_file)}})
            if get != None:
                db.delete(ids=get["ids"])



    def chat(self, command):

        

        db = Chroma(persist_directory="./chroma.db", embedding_function=self.embedding)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        if BASETEN == "1":
            #prompt_template = self.get_prompt(self.INSTRUCTION, self.SYSTEM_PROMPT)
            prompt_template = self.wizard_prompt
            llama_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain_type_kwargs = {"prompt": llama_prompt}
            qa_interface = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=False
            )

        else:

            qa_interface = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
        response = qa_interface(command)
        self.answer = response["result"]
        return self.answer
    
    def get_documents(self) -> set:

        doc_list = set()
        db = Chroma(persist_directory="./chroma.db", embedding_function=self.embedding)
        docs = db.get()
        metas = docs["metadatas"]

        for meta in metas:

            doc_list.add(meta["source"])

        return doc_list


class MyApp(App):
    CSS_PATH = "style.tcss"

    def is_decimal(self, string):
        try:
            Decimal(string)
            return True
        except InvalidOperation:
            return False

    def compose(self) -> ComposeResult:

        self.embedding = None
        match EMBEDDING_FUNCTION:
            case "openai":
                self.embedding = OpenAIEmbeddings()
            case "huggingface":
                self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

        if (BASETEN == "1"):
            self.ai = AI(baseten_api_key=BASETEN_API_KEY, baseten_id=BASETEN_ID,temperature=0.01,tokens=1024, embedding=self.embedding)
        else:
            self.ai = AI(OPENAI_API_KEY, temperature=0.1, model="gpt-3.5-turbo", tokens=1024, embedding=self.embedding)
        self.serper = Serper(SERPER_API_KEY, self.ai.llm)
        self.source = "file"
        

       
        yield Header(id="header")
        
        with TabbedContent(initial="chat_tab", id="tabbed_content"):
            with TabPane("Chat", id="chat_tab"):

                yield Input(placeholder="Question", id="chat_input")
                
                yield Markdown("", id="chat_out")
                
            with TabPane("Settings", id="setting_tab"):


                yield Label("Source: ")
                
                with RadioSet(id="settings_radioset"):
                    yield RadioButton("Uploaded files", id="radiobtn_files", value=True)
                    yield RadioButton("Google", id="radiobtn_google")
                    yield RadioButton("ChatGPT", id="radiobtn_chatgpt")

                
                yield Horizontal(
                    Button("Temperature (0-1)", id="settings_label", disabled=True),
                    Input(
                        name="Temperature",
                        value=str(self.ai.temperature), 
                        id="settings_input")
                )
                
                
                yield Button.success("Save", id="settings_btn")


            with TabPane("Upload", id="upload"):
               
                yield DirectoryTree(Path.home(), id="directorytree")
                
                yield Button.success("Upload", id="upload_btn")
            
            with TabPane("Files", id="myfiles"):

                yield Button.error("Remove file", id="remove_btn")

                yield ListView(id="file_listview")

                
            
            with TabPane("Log", id="log_tab"):
                yield Log()
            

        yield Footer()

    @on(RadioSet.Changed)
    def update_source(self, event: RadioSet.Changed) -> None:
        log = self.query_one(Log)
        match event.radio_set.pressed_button.id:
            case "radiobtn_google":
                self.source = "google"
                log.write_line("[+] Source: google")
            case "radiobtn_files":
                self.source = "file"
                log.write_line("[+] Source: local files")
            case "radiobtn_chatgpt":
                self.source = "chatgpt"
                log.write_line("[+] Source: ChatGPT")


    @on(Input.Submitted)
    def update_text(self, event: Input.Submitted) -> None:
        match event.input.id:
            case "chat_input":
                match self.source:
                    case "file":
                        result = self.ai.chat(event.value)
                    case "google":
                        result = self.serper.chat(event.value)

                self.query_one("#chat_out", Markdown).update(result)

    @on(DirectoryTree.FileSelected)
    def select_file(self, event: DirectoryTree.FileSelected) -> None:
        self.ai.file = event.path.as_posix()     

    @on(Button.Pressed)
    def upload_file(self, event: Button.Pressed) -> None:
        log = self.query_one(Log)
        match event.button.id:
            case "upload_btn":
                if (self.ai.upload()):
                    log.write_line("[+] Document was uploaded successfully")
                else:
                    log.write_line("[-] Upload failed. Only PDF and txt documents are supported")
                self.show_documents()

            case "remove_btn":
                self.ai.remove()
                self.show_documents()
                
            case "settings_btn":
                temperature = self.query_one("#settings_input", Input).value
                if self.is_decimal(temperature):
                    if (float(temperature) >= 0) and (float(temperature) <=1):
                        self.ai.set_temperature(temperature)
                        log.write_line("[+] Temperature: " + temperature)
                    else:
                        log.write_line("[!] Temperature is not between 0 and 1.")
                else:
                    log.write_line("[!] Temperature is not a decimal number.")

    @on(ListView.Selected)
    def select_document(self, event: ListView.Selected):
        log = self.query_one(Log)
        log.write_line(str(event.item.name))
        self.ai.remove_file = str(event.item.name)

    @on(TabbedContent.TabActivated)
    def refresh_directorytree(self, event: TabbedContent.TabActivated):
        match event.tab.id:
            case "upload":
                self.query_one("#directorytree", DirectoryTree).reload()

    def show_documents(self):

        docs = self.ai.get_documents()
        list_items = self.query(ListItem)
        for item in list_items:
            item.remove()
        for doc in docs:
            
            self.query_one("#file_listview", ListView).append(ListItem(Label(str(doc)), name=str(doc)))
          

    def on_mount(self) -> None:
        self.title = "FoxyIntel v0.1"

        self.show_documents()
        
        
if __name__=="__main__":
    app = MyApp()
    app.run()
