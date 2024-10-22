import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")

with st.form("my_form"):
    st.title("Sentiment Analysis")
    text_review = st.text_area("Write me a review")

    option = st.selectbox(
        "Select the language to evaluate:", ("Italian", "Spanish", "English")
    )
    submitted = st.form_submit_button("Submit")
    if submitted:

        # 1 prompt template
        template1 = """
        Please act as a machine learning model trained for perform a supervised learning task,
        for extract the sentiment of a review in '{option}' Language.

        Give your answer writing a Json evaluating the sentiment field between the dollar sign, the value must be printed without dollar sign.
        The value of sentiment must be "positive"  or "negative", otherwise if the text is not valuable write "null".

        Example:

        field 1 named :
        text_review with value: {text_review}
        field 2 named :
        sentiment with value: $sentiment$
        Field 3 named :
        language with value: {option}
        Review text: '''{text_review}'''

        """

        # 2 prompt template
        template2 = """
        Please act as a machine learning model trained for perform a supervised learning task,
        for extract the outcome of a call conversation in '{option}' Language. Please Determine call outcome either "issue resolved", or "follow-up action needed".

        Example:

        field 1 named :
        text_review with value: {text_review}
        field 2 named :
        outcome with value: $outcome$
        Field 3 named :
        language with value: {option}
        Review text: '''{text_review}'''

        """

        prompt = PromptTemplate(
            template=template2, input_variables=["text_review", "option"]
        )

        llm_chain = LLMChain(prompt=prompt, llm=llm)

        if prompt:
            response = llm_chain.run({"text_review": text_review, "option": option})
            print(response)
            st.text(response)
