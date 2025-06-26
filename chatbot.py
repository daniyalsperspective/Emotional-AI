from transformers import pipeline
import gradio as gr

emotion_classifier = pipeline("text-classification", 
                              model="bhadresh-savani/distilbert-base-uncased-emotion")

def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result['label'], result['score']

def generate_response(user_input):
    emotion, _ = detect_emotion(user_input)
    responses = {
        "joy": "I'm glad to hear you're feeling happy! 😊",
        "anger": "I can see you're upset. Let's talk about it. 🧘",
        "sadness": "I'm here for you. It’s okay to feel this way. 💙",
        "fear": "That sounds scary. Want to share more? 🤝",
        "love": "That’s heartwarming! 💖",
        "surprise": "Wow! That’s unexpected. Tell me more! 😮"
    }
    reply = responses.get(emotion, "I’m listening... Tell me more.")
    full_reply = f"{reply}\n\n🧠 Detected Emotion: *{emotion}*"
    return full_reply, emotion

def chatbot(user_input, history):
    if history is None:
        history = []
    reply, emotion = generate_response(user_input)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return history, history

def reset():
    return [], []

with gr.Blocks(title="Empathetic AI Chatbot") as demo:
    gr.Markdown("## 🤖 Empathetic AI Chatbot with Emotion Detection")
    chatbot_output = gr.Chatbot(label="Chat", type="messages")
    user_input = gr.Textbox(label="Your Message", placeholder="Type here...", lines=2)
    state = gr.State()
    with gr.Row():
        send_btn = gr.Button("Send")
        reset_btn = gr.Button("🔁 Reset Chat")
    send_btn.click(fn=chatbot, inputs=[user_input, state], outputs=[chatbot_output, state])
    reset_btn.click(fn=reset, inputs=[], outputs=[chatbot_output, state])

demo.launch()
