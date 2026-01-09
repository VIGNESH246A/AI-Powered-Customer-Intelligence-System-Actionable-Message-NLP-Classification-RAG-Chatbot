#!/usr/bin/env python3
"""
Enhanced Interactive Streamlit Application
Integrated Customer Support System with Advanced Features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from src.rag_system.vector_store import load_vector_store
from src.integration.unified_pipeline import create_unified_pipeline
from src.rag_system.config import APP_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
/* Main styling */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #1f77b4, #4caf50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
    animation: fadeIn 1s;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Chat messages */
.chat-message {
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    animation: slideIn 0.3s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #2196f3;
}

.assistant-message {
    background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
    border-left: 5px solid #4caf50;
}

.non-actionable-message {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-left: 5px solid #ff9800;
}

/* Classification badge */
.classification-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
    margin-bottom: 1rem;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

.actionable-badge {
    background: linear-gradient(135deg, #4caf50, #45a049);
    color: white;
}

.non-actionable-badge {
    background: linear-gradient(135deg, #ff9800, #f57c00);
    color: white;
}

/* Stats cards */
.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin: 0.5rem 0;
}

.stat-value {
    font-size: 2.5rem;
    font-weight: bold;
    margin: 0.5rem 0;
}

.stat-label {
    font-size: 1rem;
    opacity: 0.9;
}

/* Context box */
.context-box {
    background: linear-gradient(135deg, #fff3cd 0%, #ffe9a7 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid #ffc107;
    margin-top: 1rem;
}

.context-chunk {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid #ddd;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Sidebar */
.sidebar-content {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Progress bar */
.confidence-bar {
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
    height: 8px;
    border-radius: 4px;
    margin: 0.5rem 0;
}

/* Quick action buttons */
.quick-action {
    background: #f0f2f6;
    padding: 0.8rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    cursor: pointer;
    transition: all 0.3s;
    border: 1px solid #ddd;
}

.quick-action:hover {
    background: #e1e4e8;
    transform: translateX(5px);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load unified pipeline (cached)"""
    try:
        vector_store = load_vector_store()
        classifier_model = st.session_state.get('classifier_model', 'random_forest')
        pipeline = create_unified_pipeline(vector_store, classifier_model=classifier_model)
        return pipeline
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        st.info("💡 Please run 'python build_index.py' first to create the vector index.")
        return None


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stats" not in st.session_state:
        st.session_state.stats = {
            'total_messages': 0,
            'actionable_count': 0,
            'non_actionable_count': 0,
            'avg_confidence': 0
        }
    if "classifier_model" not in st.session_state:
        st.session_state.classifier_model = 'random_forest'


def update_stats(classification):
    """Update conversation statistics"""
    st.session_state.stats['total_messages'] += 1
    
    if classification['is_actionable']:
        st.session_state.stats['actionable_count'] += 1
    else:
        st.session_state.stats['non_actionable_count'] += 1
    
    # Update average confidence
    total = st.session_state.stats['total_messages']
    current_avg = st.session_state.stats['avg_confidence']
    new_confidence = classification['confidence']
    st.session_state.stats['avg_confidence'] = (current_avg * (total - 1) + new_confidence) / total


def create_confidence_gauge(confidence, is_actionable):
    """Create an interactive confidence gauge"""
    color = '#4caf50' if is_actionable else '#ff9800'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff3e0'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'size': 14}
    )
    
    return fig


def create_stats_chart():
    """Create statistics chart"""
    stats = st.session_state.stats
    
    if stats['total_messages'] == 0:
        return None
    
    labels = ['Actionable', 'Non-Actionable']
    values = [stats['actionable_count'], stats['non_actionable_count']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#4caf50', '#ff9800']),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Message Classification Distribution",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True
    )
    
    return fig


def display_message(role, content, classification=None):
    """Display enhanced chat message"""
    if role == "user":
        st.markdown(
            f'<div class="chat-message user-message">'
            f'<b>👤 You</b><br><br>{content}'
            f'</div>', 
            unsafe_allow_html=True
        )
    
    elif role == "assistant":
        if classification:
            is_actionable = classification.get('is_actionable', False)
            confidence = classification.get('confidence', 0)
            method = classification.get('method', 'unknown')
            
            # Badge
            badge_class = "actionable-badge" if is_actionable else "non-actionable-badge"
            badge_text = "✅ Actionable" if is_actionable else "ℹ️ Non-Actionable"
            
            st.markdown(
                f'<div class="classification-badge {badge_class}">'
                f'{badge_text} | Confidence: {confidence:.1%} | Method: {method}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Message
            message_class = "assistant-message" if is_actionable else "non-actionable-message"
            icon = "🤖" if is_actionable else "💬"
            
            st.markdown(
                f'<div class="chat-message {message_class}">'
                f'<b>{icon} Assistant</b><br><br>{content}'
                f'</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">'
                f'<b>🤖 Assistant</b><br><br>{content}'
                f'</div>', 
                unsafe_allow_html=True
            )


def show_quick_actions():
    """Display quick action buttons"""
    st.markdown("### 🚀 Quick Test Messages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Actionable Examples:**")
        actionable_msgs = [
            "What is the refund policy?",
            "My order hasn't arrived",
            "How do I reset password?",
            "The product is damaged",
            "Need help with installation"
        ]
        
        for msg in actionable_msgs:
            if st.button(f"📝 {msg}", key=f"action_{msg}", use_container_width=True):
                return msg
    
    with col2:
        st.markdown("**Non-Actionable Examples:**")
        non_actionable_msgs = [
            "Thank you for your help!",
            "I love this product!",
            "Amazing experience!",
            "Great service!",
            "Fast delivery!"
        ]
        
        for msg in non_actionable_msgs:
            if st.button(f"💬 {msg}", key=f"non_action_{msg}", use_container_width=True):
                return msg
    
    return None


def main():
    """Main application"""
    
    # Initialize
    init_session_state()
    
    # Header with animation
    st.markdown(f'<div class="main-header">🤖 {APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">✨ Intelligent Customer Support with AI-Powered Classification & RAG</div>',
        unsafe_allow_html=True
    )
    
    # Load pipeline
    pipeline = load_pipeline()
    
    if pipeline is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Customer+Support+AI", use_container_width=True)
        
        st.markdown("---")
        st.header("⚙️ Configuration")
        
        # Model selection with visual indicator
        st.subheader("🧠 Classifier Model")
        classifier_choice = st.radio(
            "Choose classification model:",
            ["🌲 Random Forest (Fast)", "🧬 LSTM (Accurate)"],
            key="classifier_radio"
        )
        
        model_type = 'random_forest' if "Random Forest" in classifier_choice else 'lstm'
        
        if st.session_state.classifier_model != model_type:
            st.session_state.classifier_model = model_type
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Display options
        st.subheader("🎨 Display Options")
        show_classification = st.checkbox("📊 Show Classification Details", value=True)
        show_confidence_gauge = st.checkbox("🎯 Show Confidence Gauge", value=True)
        show_context = st.checkbox("📚 Show Retrieved Context", value=False)
        show_scores = st.checkbox("🔢 Show Relevance Scores", value=False)
        show_method = st.checkbox("🔍 Show Detection Method", value=True)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📈 Session Statistics")
        stats = st.session_state.stats
        
        if stats['total_messages'] > 0:
            # Create metric cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Total Messages",
                    stats['total_messages'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Avg Confidence",
                    f"{stats['avg_confidence']:.1%}",
                    delta=None
                )
            
            st.metric(
                "Actionable",
                stats['actionable_count'],
                delta=f"{stats['actionable_count']/stats['total_messages']*100:.0f}%"
            )
            
            st.metric(
                "Non-Actionable",
                stats['non_actionable_count'],
                delta=f"{stats['non_actionable_count']/stats['total_messages']*100:.0f}%"
            )
            
            # Show chart
            chart = create_stats_chart()
            if chart:
                st.plotly_chart(chart, use_container_width=True, key="sidebar_stats_chart")
        else:
            st.info("📊 Start chatting to see statistics!")
        
        st.markdown("---")
        
        # System info
        if st.button("ℹ️ System Information", use_container_width=True):
            sys_stats = pipeline.get_stats()
            st.json(sys_stats)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True, type="primary"):
            pipeline.clear_conversation()
            st.session_state.messages = []
            st.session_state.stats = {
                'total_messages': 0,
                'actionable_count': 0,
                'non_actionable_count': 0,
                'avg_confidence': 0
            }
            st.success("✅ Conversation cleared!")
            st.rerun()
    
    # Main content area
    tabs = st.tabs(["💬 Chat", "🚀 Quick Actions", "📊 Analytics", "ℹ️ Help"])
    
    # Tab 1: Chat
    with tabs[0]:
        # Display chat history
        for message in st.session_state.messages:
            display_message(
                message["role"], 
                message["content"],
                message.get("classification")
            )
            
            # Show confidence gauge
            if show_confidence_gauge and message.get("classification"):
                classification = message["classification"]
                gauge = create_confidence_gauge(
                    classification['confidence'],
                    classification['is_actionable']
                )
                # Use unique key based on message index
                msg_idx = st.session_state.messages.index(message)
                st.plotly_chart(gauge, use_container_width=True, key=f"gauge_{msg_idx}")
            
            # Show context if enabled
            if show_context and message.get("rag_context"):
                with st.expander("📚 Retrieved Context", expanded=False):
                    st.markdown('<div class="context-box">', unsafe_allow_html=True)
                    
                    context_chunks = message["rag_context"].get("context_chunks", [])
                    scores = message.get("scores", [])
                    
                    for idx, chunk in enumerate(context_chunks, 1):
                        st.markdown(f"**📄 Context {idx}** (Section: {chunk.get('section', 'N/A')})")
                        
                        with st.container():
                            st.text_area(
                                f"Content {idx}",
                                chunk['text'],
                                height=150,
                                key=f"context_{idx}_{hash(chunk['text'])}",
                                disabled=True
                            )
                        
                        if show_scores and scores and idx-1 < len(scores):
                            st.progress(scores[idx-1])
                            st.caption(f"Relevance Score: {scores[idx-1]:.4f}")
                        
                        if idx < len(context_chunks):
                            st.markdown("---")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        prompt = st.chat_input("💬 Type your message here...", key="chat_input")
        
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process message
            with st.spinner("🤔 Processing your message..."):
                if show_scores:
                    result = pipeline.process_message_with_scores(prompt)
                else:
                    result = pipeline.process_message(prompt)
            
            # Update statistics
            update_stats(result['classification'])
            
            # Prepare message data
            message_data = {
                "role": "assistant",
                "content": result['response'],
                "classification": result['classification']
            }
            
            if result['is_actionable'] and result['rag_context']:
                message_data["rag_context"] = result['rag_context']
                
                if show_scores and 'relevance_scores' in result['rag_context']:
                    message_data["scores"] = result['rag_context']['relevance_scores']
            
            st.session_state.messages.append(message_data)
            st.rerun()
    
    # Tab 2: Quick Actions
    with tabs[1]:
        st.markdown("### 🚀 Test the System with Pre-Made Messages")
        st.markdown("Click any button below to send a test message:")
        
        selected_msg = show_quick_actions()
        
        if selected_msg:
            st.session_state.messages.append({"role": "user", "content": selected_msg})
            
            with st.spinner("🤔 Processing..."):
                if show_scores:
                    result = pipeline.process_message_with_scores(selected_msg)
                else:
                    result = pipeline.process_message(selected_msg)
            
            update_stats(result['classification'])
            
            message_data = {
                "role": "assistant",
                "content": result['response'],
                "classification": result['classification']
            }
            
            if result['is_actionable'] and result['rag_context']:
                message_data["rag_context"] = result['rag_context']
                if show_scores and 'relevance_scores' in result['rag_context']:
                    message_data["scores"] = result['rag_context']['relevance_scores']
            
            st.session_state.messages.append(message_data)
            st.rerun()
    
    # Tab 3: Analytics
    with tabs[2]:
        st.markdown("### 📊 Conversation Analytics")
        
        if len(st.session_state.messages) == 0:
            st.info("📭 No messages yet. Start chatting to see analytics!")
        else:
            # Create analytics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-value">{stats["total_messages"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">Total Messages</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-value">{stats["actionable_count"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">Actionable</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-value">{stats["non_actionable_count"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">Non-Actionable</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-value">{stats["avg_confidence"]:.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">Avg Confidence</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show detailed breakdown
            if stats['total_messages'] > 0:
                chart = create_stats_chart()
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key="analytics_pie_chart")
                
                # Message timeline
                st.markdown("### 📅 Message Timeline")
                
                # Extract data
                message_data = []
                for idx, msg in enumerate(st.session_state.messages):
                    if msg['role'] == 'assistant' and 'classification' in msg:
                        message_data.append({
                            'Message': idx // 2 + 1,
                            'Type': 'Actionable' if msg['classification']['is_actionable'] else 'Non-Actionable',
                            'Confidence': msg['classification']['confidence']
                        })
                
                if message_data:
                    df = pd.DataFrame(message_data)
                    
                    fig = px.line(
                        df,
                        x='Message',
                        y='Confidence',
                        color='Type',
                        markers=True,
                        title='Confidence Over Time'
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="analytics_timeline_chart")
    
    # Tab 4: Help
    with tabs[3]:
        st.markdown("### ℹ️ How to Use This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 Features
            - **Automatic Classification**: Messages are classified as actionable or non-actionable
            - **Smart Routing**: Actionable messages get RAG-powered responses
            - **Confidence Scores**: See how confident the system is
            - **Context Retrieval**: View the knowledge base sources used
            - **Analytics**: Track your conversation patterns
            """)
        
        with col2:
            st.markdown("""
            #### 🚀 Quick Start
            1. Type your message in the chat input
            2. System automatically classifies it
            3. Get appropriate response
            4. View confidence and context (if enabled)
            5. Check analytics for insights
            """)
        
        st.markdown("---")
        
        st.markdown("""
        #### 📝 Message Examples
        
        **Actionable Messages** (will use RAG chatbot):
        - "What is the refund policy?"
        - "My order hasn't arrived yet"
        - "How do I reset my password?"
        - "The product is damaged"
        - "I need help with installation"
        
        **Non-Actionable Messages** (acknowledgment only):
        - "Thank you for your help!"
        - "I love this product!"
        - "Amazing experience!"
        - "Great service!"
        - "Fast delivery!"
        """)
        
        st.markdown("---")
        
        st.markdown("""
        #### ⚙️ Configuration Options
        - **Classifier Model**: Choose between Random Forest (fast) or LSTM (accurate)
        - **Display Options**: Customize what information to show
        - **Quick Actions**: Use pre-made test messages
        - **Analytics**: View conversation statistics and patterns
        """)


if __name__ == "__main__":
    main()