from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
from decimal import Decimal, InvalidOperation
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
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
from textual.widgets.selection_list import Selection
from textual.widgets._tabbed_content import ContentTab

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')



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

    def __init__(self, api_key, temperature, model, tokens):
        self.api_key = api_key
        self.temperature = temperature
        self.tokens = tokens
        self.model = model
        self.llm = OpenAI(openai_api_key=api_key, temperature=temperature, model=self.model, max_tokens=self.tokens)
        self.answer = None
        self.file = None
        self.remove_file = None
    def set_temperature(self, temperature):
        self.temperature = float(temperature)
        self.llm = OpenAI(openai_api_key=self.api_key, temperature=self.temperature, model=self.model, max_tokens=self.tokens)
    
    def upload(self):
        raw_doc = TextLoader(self.file).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(raw_doc)
        db = Chroma.from_documents(docs, OpenAIEmbeddings(),persist_directory="./chroma.db")

    def remove(self):

        if self.remove_file != None:
            db = Chroma(persist_directory="./chroma.db", embedding_function=OpenAIEmbeddings())
            
            get = db.get(include=["metadatas"], where={"source":{"$eq":str(self.remove_file)}})
            if get != None:
                db.delete(ids=get["ids"])



    def chat(self, command):

        db = Chroma(persist_directory="./chroma.db", embedding_function=OpenAIEmbeddings())

        vectorstore_info = VectorStoreInfo(
            name="test_db",
            description="testing storing information to vector store",
            vectorstore=db,
        )

        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
        agent_executor = create_vectorstore_agent(llm=self.llm, toolkit=toolkit, verbose=False)
        self.answer = agent_executor.run(command)
        return self.answer
    
    def get_documents(self):

        doc_list = set()
        db = Chroma(persist_directory="./chroma.db", embedding_function=OpenAIEmbeddings())
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
        self.ai = AI(OPENAI_API_KEY, 0.1, model="text-davinci-003", tokens=512)
        self.serper = Serper(SERPER_API_KEY, self.ai.llm)
        self.source = "file"
        

       
        yield Header(id="header")
        
        with TabbedContent(initial="chat_tab", id="tabbed_content"):
            with TabPane("Chat", id="chat_tab"):

                yield Input(placeholder="Question", id="chat_input")
                
                yield Markdown("", id="chat_out")
                
            with TabPane("Settings", id="setting_tab"):

                yield Horizontal(
                    Button("Temperature", id="settings_label", disabled=True),
                    Input(
                        name="Temperature",
                        value=str(self.ai.temperature), 
                        id="settings_input")
                )
                with RadioSet(id="settings_radioset"):
                    yield RadioButton("Uploaded files", id="radiobtn_files", value=True)
                    yield RadioButton("Google", id="radiobtn_google")
                    yield RadioButton("ChatGPT", id="radiobtn_chatgpt")
                
                yield Button.success("Save", id="settings_btn")
                
            with TabPane("Upload", id="upload"):
               
                yield DirectoryTree(Path.home())
                
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
                self.ai.upload()
                log.write_line("[+] Document was uploaded")
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



    def show_documents(self):

        docs = self.ai.get_documents()
        list_items = self.query(ListItem)
        for item in list_items:
            item.remove()
        for doc in docs:
            
            self.query_one("#file_listview", ListView).append(ListItem(Label(str(doc)), name=str(doc)))
          

    def on_mount(self) -> None:
        self.title = "FoxyIntel v.0.1"

        self.show_documents()
        
        
if __name__=="__main__":
    app = MyApp()
    app.run()
