# End to End Evaluation Template for RAG Apps
This is a template and you add or remove metrics based on your usecases. Also there are `LLMasJudge` abstract classes which are useful to use [PHUDGE](https://github.com/deshwalmahesh/PHUDGE/) and GPT4 type of LLMs as evaluator. For more details to understand the evaluations, look inside `SEE_THIS.ipynb`

![Alt Text](./APP.png)


# How to use
Tested with: `Python 3.9`

Step: 
1. `pip install -r requirements.txt`
2. `pip install -U evaluate` (without it, some old metrics won't work)
3. `streamlit run eval_rag_app.py`
