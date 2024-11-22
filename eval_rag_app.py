import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from eval_metrics import * 
from inspeq.client import InspeqEval

if 'api_key' not in st.session_state: st.session_state['api_key'] = None
if 'INSPEQ_API_KEY' not in st.session_state: st.session_state['INSPEQ_API_KEY'] = None
if 'INSPEQ_PROJECT_ID' not in st.session_state: st.session_state['INSPEQ_PROJECT_ID'] = None
if 'user_turn' not in st.session_state: st.session_state['user_turn'] = False
if 'pdf' not in st.session_state: st.session_state['pdf'] = None
if "embed_model" not in st.session_state: st.session_state['embed_model'] = None
if "vector_store" not in st.session_state: st.session_state['vector_store'] = None
if "metrics" not in st.session_state: st.session_state['metrics'] = None
if "eval_models" not in st.session_state: st.session_state["eval_models"] = {"app_metrics": AppMetrics()}
if "options" not in st.session_state: st.session_state['options'] = []

st.set_page_config(page_title="Document Genie", layout="wide")

def get_inspeq_evaluation(prompt, response, context, metric):
    inspeq_eval = InspeqEval(inspeq_api_key=st.session_state['INSPEQ_API_KEY'], inspeq_project_id= st.session_state['INSPEQ_PROJECT_ID'])
    input_data = [{
    "prompt": prompt,
    "response": response,
    "context": context
        }]
    metrics_list = metric
    try:
        output = inspeq_eval.evaluate_llm_task(
            metrics_list=metrics_list,
            input_data=input_data,
            task_name="task"
        )
        return output
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


@AppMetrics.measure_execution_time
def build_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = st.session_state['chunk_size'] , chunk_overlap= st.session_state['chunk_overlap'])
    text_chunks = text_splitter.split_text(text)
    st.session_state['vector_store']= LanceDB.from_texts(text_chunks, st.session_state["embed_model"])

@AppMetrics.measure_execution_time
def fetch_context(query):
    return st.session_state['vector_store'].similarity_search(query, k = st.session_state['top_k'])

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I don't think the answer is available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=st.session_state['api_key'])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@AppMetrics.measure_execution_time
def llm_output(chain, docs, user_question):
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)


def user_input(user_question):
    contexts_with_scores, exec_time = fetch_context(user_question)
    st.session_state["eval_models"]["app_metrics"].exec_times["chunk_fetch_time"] = exec_time

    chain = get_conversational_chain()
    response, exec_time = llm_output(chain, contexts_with_scores, user_question)
    st.session_state["eval_models"]["app_metrics"].exec_times["llm_resp_time"] = exec_time
    
    st.write("Reply: ", response["output_text"])

    ctx = ""
    for item in contexts_with_scores:
        if len(item.page_content.strip()):
            ctx += f"<br>{item.page_content}<br>&nbsp</li>"

    with st.expander("Click to see the context passed"):
        st.markdown(f"""<ol>{ctx}</ol>""", unsafe_allow_html=True)
    
    return contexts_with_scores, response["output_text"]

def result_to_list(results):
    eval = []
    score = []
    label = []

    return eval, score, label
def evaluate_all(query, context_lis, response, metrics_list):

    context = "\n\n".join(context_lis) if len(context_lis) else "no context"
    
    RESULT = {}

    RESULT["guards"] = {
        "evaluations" : get_inspeq_evaluation(query, response, context, metrics_list)
    }
    RESULT["execution_times"] = (st.session_state["eval_models"]["app_metrics"].exec_times)
    
    return RESULT


def main():
    st.markdown("""## Inspeq RAG Demo""")
    

    with st.expander("### 📝 Usage Note"):
        st.markdown("""
        **Key Features**:
        1. **Document Upload**: Easily upload and analyze PDF documents for text-based insights.
        2. **Vector Store Creation**: Build efficient and searchable vector stores with LanceDB for enhanced data retrieval.
        3. **Question Answering**: Ask in-depth questions related to your uploaded PDFs, with GPT-3.5 providing detailed, context-aware answers.
        4. **Inspeq Metrics Evaluation**: Use the Inspeq API to assess and monitor the performance of language models (LLMs) based on customized metrics.
        """)

    with st.expander("### 🛠️ How to Use"):
        st.markdown("""
        1. Enter your **OpenAI** and **Inspeq API Keys**.
        2. Upload PDF files to analyze.
        3. Adjust chunk size, overlap, and top-K contexts for customization.
        4. Ask questions and select metrics for evaluation.
        5. View the response and evaluation results from the Inspeq SDK in a detailed table.
        """)

    
    with st.sidebar:
        st.title("Menu:")
        st.session_state['api_key'] = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
        st.session_state['INSPEQ_API_KEY'] =  st.text_input("Enter your inspeq API key", type="password", key="inspeq_api_key")
        st.session_state['INSPEQ_PROJECT_ID'] =  st.text_input("Enter your inspeq project ID:", type="password", key="inspeq_project_id")

        _ =  st.number_input("Top-K Contxets to fetch", min_value=1, max_value=50, value=3, step=1, key="top_k")
        _ = st.number_input("Chunk Length", min_value=8, max_value=4096, value=512, step=8, key="chunk_size")
        _ = st.number_input("Chunk Overlap Length", min_value=4, max_value=2048, value=64, step=1, key="chunk_overlap")

        st.session_state["pdf"] = st.file_uploader("Upload your PDF Files...", accept_multiple_files=True, key="pdf_uploader")

        if st.session_state["pdf"]:
            if st.session_state["embed_model"] is None: 
                with st.spinner("Setting up `all-MiniLM-L6-v2` for the first time"):
                    st.session_state["embed_model"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # with st.spinner("Processing PDF files..."):
                raw_text = get_pdf_text(st.session_state["pdf"])
            # with st.spinner("Creating `LanceDB` Vector stores from texts..."):
                _, exec_time = build_vector_store(raw_text)
                st.session_state["eval_models"]["app_metrics"].exec_times["chunk_creation_time"] = exec_time
                st.success("Done")

    if not (st.session_state['api_key'] and st.session_state['INSPEQ_API_KEY'] and st.session_state['INSPEQ_PROJECT_ID']) :
        st.warning("Enter OpenAI API Key, Inspeq API key and Inspeq Project ID to proceed")
    elif not st.session_state["pdf"]:
        st.warning("Upload a PDF file")
    else:
        st.markdown("""#### Ask a Question from the PDF file""")
        if "options" not in st.session_state:  # Ensure session state is initialized
            st.session_state["options"] = []
        # Form for asking questions and evaluating
        with st.form(key="my_form"):
            user_question = st.text_input("Enter your question:", key="user_question")  # Move input inside the form
            
            list_of_metrics = ["RESPONSE_TONE", "ANSWER_RELEVANCE", "FACTUAL_CONSISTENCY", "CONCEPTUAL_SIMILARITY", "READABILITY", "COHERENCE", "CLARITY", 
                               "DIVERSITY", "CREATIVITY", "NARRATIVE_CONTINUITY", "GRAMMATICAL_CORRECTNESS", "PROMPT_INJECTION", 
                               "DATA_LEAKAGE", "INSECURE_OUTPUT", "INVISIBLE_TEXT", "TOXICITY", "BLEU_SCORE", "COMPRESSION_SCORE", 
                               "COSINE_SIMILARITY_SCORE", "FUZZY_SCORE", "METEOR_SCORE", "ROUGE_SCORE"]
            selected_metrics = st.multiselect(
                "Select the metrics to Evaluate",
                list_of_metrics,
                default=st.session_state["options"]
            )
            submit_button = st.form_submit_button(label="Generate and Evaluate")  # Form submission button

        if submit_button:  # Only process when Evaluate button is pressed
            if not selected_metrics:
                st.session_state["options"] = []  # Reset session state if no metrics are selected
            else:
                st.session_state["options"] = selected_metrics 
            if user_question and st.session_state['api_key']:  # Ensure question and API key are provided
                with st.spinner("Getting Response from LLM..."):
                    contexts_with_scores, response = user_input(user_question)

                with st.spinner("Calculating all the matrices. Please wait..."):
                    selected = []
                    for i in range(len(st.session_state["options"])):
                        selected.append(st.session_state["options"][i])
                    
                    eval_result = evaluate_all(user_question, [item.page_content for item in contexts_with_scores], response, selected)

                with st.expander("Click to see all the evaluation metrics"):
                    st.json(eval_result)
                    error_logs = []
                    metric_name = []
                    eval = []
                    score = []
                    label = []
                    for i in range(len(eval_result["guards"]["evaluations"]["results"])):
                        eval_status =  eval_result["guards"]["evaluations"]["results"][i]["metric_evaluation_status"]
                        name = eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["metric_name"]
                        new_name = name.replace("_EVALUATION", "")
                        if eval_status != "EVAL_FAIL":
                            metric_name.append(new_name)
                            eval.append(eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["threshold"][0])
                            score.append(eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["actual_value"])
                            label.append(eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["metric_labels"][0])
                        else:
                            error_logs.append(eval_result["guards"]["evaluations"]["results"][i]["error_message"])
                    st.write("The labels are ", eval_result["guards"]["evaluations"]["results"][i]["evaluation_details"]["metric_labels"])

                final_result = {
                    "Metric": metric_name,
                    "Evaluation Result": eval,
                    "Score": [round(x, 2) for x in score],
                    "Label": label
                }
                df = pd.DataFrame(final_result)
                st.table(df)
                # If there are any errors when accessing the API
                if error_logs:
                    with st.expander("Error Logs"):
                        for error in error_logs:
                            st.write(error)
                st.session_state["options"] = []    # reinitialise the state to blank after the evaluation is done

if __name__ == "__main__":
    main()
