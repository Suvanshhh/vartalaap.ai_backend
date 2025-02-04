def generate_support_context(sentiment_result, classification_result):
    """Generate context for the Gemma model based on analysis"""
    context = f"""
    This is a customer support inquiry with {sentiment_result['sentiment']} sentiment 
    and {sentiment_result['priority']} priority. 
    Category: {classification_result['primary_category']['category']}
    
    Respond in a {sentiment_result['sentiment']} and professional tone.
    """
    
    if sentiment_result['needs_escalation']:
        context += "\nThis issue requires escalation to a human agent."
    if sentiment_result['is_urgent']:
        context += "\nThis is an urgent matter requiring immediate attention."
    
    return context

def format_bot_response(bot_response, sentiment_result):
    """Format the bot response based on analysis"""
    response = bot_response.strip()
    
    if sentiment_result['needs_escalation']:
        response += "\n\nI'll connect you with a customer support representative who can better assist you with this matter."
    elif sentiment_result['is_urgent']:
        response += "\n\nI understand this is urgent. I'll prioritize your request."
        
    return response