template = """
    Please act as a machine learning model trained to perform a supervised learning task,
    extracting the sentiment of provided call conversation text {text_review}
    from a customer to a staff member in the '{team}' team of
    "Customer Support", "PA Agent", "Technical Support", or "Unknown".

    Provide your answer response in JSON format,
    evaluating the sentiment field between the dollar signs.
    The value must be printed without dollar signs.
    in the answer response, provide team with value {team}, sentiment, sentiment_score,
    outcome, and summary fields.
    The sentiment value must be one of "positive", "neutral", or "negative".
    Also, provide the sentiment analysis score between 0 and 1.
    Determine the call outcome as either "issue resolved" or "follow-up action needed".
    Additionally, provide a summary of the conversation in the review text field.
    The summary should be a short description of the conversation, less than 30 words.
    """
