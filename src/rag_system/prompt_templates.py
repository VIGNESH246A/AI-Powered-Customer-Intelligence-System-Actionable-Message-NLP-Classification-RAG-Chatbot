"""Prompt templates for RAG system"""

SYSTEM_PROMPT = """You are an intelligent customer service assistant for Home Essentials, a smart home and electronics retailer.

Your responsibilities:
- Answer customer questions about products, policies, orders, and technical issues
- Provide accurate information based on the knowledge base
- Be helpful, professional, and concise
- If information is not in the provided context, politely say you don't have that information
- Always cite specific policy numbers (e.g., AP-001) or product SKUs when relevant

Guidelines:
- Use the provided context to answer questions accurately
- Don't make up information not in the context
- For technical issues, provide step-by-step troubleshooting
- For policy questions, reference the specific policy section
- Be empathetic and customer-focused
"""


def create_rag_prompt(query: str, context: str) -> str:
    """Create RAG prompt with query and context"""
    prompt = f"""{SYSTEM_PROMPT}

KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER QUESTION:
{query}

ASSISTANT RESPONSE:
"""
    return prompt


def create_followup_prompt(query: str, context: str, conversation_history: str) -> str:
    """Create prompt for follow-up questions"""
    prompt = f"""{SYSTEM_PROMPT}

CONVERSATION HISTORY:
{conversation_history}

KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER QUESTION:
{query}

ASSISTANT RESPONSE:
"""
    return prompt


# Template for specific use cases
TROUBLESHOOTING_PROMPT = """Based on the provided technical information, help the customer troubleshoot their issue.

Provide:
1. Quick diagnosis of the likely problem
2. Step-by-step solution
3. Alternative solutions if the first doesn't work
4. When to contact support (if needed)

Context: {context}
Issue: {query}
"""

PRODUCT_INFO_PROMPT = """Provide detailed product information to help the customer make an informed decision.

Include:
- Key specifications
- Features and benefits
- Pricing and warranty information
- Compatibility considerations (if applicable)

Context: {context}
Question: {query}
"""

POLICY_PROMPT = """Explain the relevant company policy clearly and accurately.

Include:
- Policy number and name
- Key terms and conditions
- Important deadlines or requirements
- Next steps for the customer

Context: {context}
Question: {query}
"""


def get_specialized_prompt(query_type: str, query: str, context: str) -> str:
    """Get specialized prompt based on query type"""
    prompts = {
        'troubleshooting': TROUBLESHOOTING_PROMPT,
        'product': PRODUCT_INFO_PROMPT,
        'policy': POLICY_PROMPT
    }
    
    template = prompts.get(query_type, create_rag_prompt(query, context))
    
    if query_type in prompts:
        return SYSTEM_PROMPT + "\n\n" + template.format(context=context, query=query)
    else:
        return template