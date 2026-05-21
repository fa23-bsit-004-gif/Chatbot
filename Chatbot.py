import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ Data
faqs = {
    "What is AI?":
    "AI means Artificial Intelligence.",

    "What is Python?":
    "Python is a programming language.",

    "Who created Python?":
    "Python was created by Guido van Rossum.",

    "What is Machine Learning?":
    "Machine Learning is a branch of AI.",

    "What is ChatGPT?":
    "ChatGPT is an AI chatbot created by OpenAI.",

    "What is Deep Learning?":
    "Deep Learning is a subset of Machine Learning.",

    "What is NLP?":
    "NLP stands for Natural Language Processing."
}

# Questions and Answers
questions = list(faqs.keys())
answers = list(faqs.values())

# Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

# Chatbot Response Function
def chatbot_response(user_input):

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    index = similarity.argmax()

    score = similarity[0][index]

    if score < 0.2:
        return "Sorry, I don't understand your question."

    return answers[index]

# Send Message Function
def send_message():

    user_message = user_entry.get()

    if user_message.strip() == "":
        return

    # Show User Message
    chat_area.config(state=tk.NORMAL)

    chat_area.insert(
        tk.END,
        "You: " + user_message + "\n",
        "user"
    )

    # Bot Response
    response = chatbot_response(user_message)

    chat_area.insert(
        tk.END,
        "Bot: " + response + "\n\n",
        "bot"
    )

    chat_area.config(state=tk.DISABLED)

    # Clear Entry
    user_entry.delete(0, tk.END)

    # Auto Scroll
    chat_area.yview(tk.END)

# Main Window
root = tk.Tk()
root.title("AI FAQ Chatbot")
root.geometry("700x600")
root.config(bg="#1e1e1e")

# Title
title = tk.Label(
    root,
    text="🤖 AI FAQ Chatbot",
    font=("Arial", 22, "bold"),
    bg="#1e1e1e",
    fg="#00ffcc"
)

title.pack(pady=10)

# Chat Area
chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=75,
    height=25,
    font=("Arial", 11),
    bg="#2b2b2b",
    fg="white",
    insertbackground="white"
)

chat_area.pack(pady=10)

chat_area.tag_config("user", foreground="#00ffcc")
chat_area.tag_config("bot", foreground="#ffcc00")

chat_area.config(state=tk.DISABLED)

# Entry Frame
entry_frame = tk.Frame(root, bg="#1e1e1e")
entry_frame.pack(pady=10)

# User Entry
user_entry = tk.Entry(
    entry_frame,
    width=45,
    font=("Arial", 14),
    bg="#2b2b2b",
    fg="white",
    insertbackground="white"
)

user_entry.grid(row=0, column=0, padx=10)

# Send Button
send_button = tk.Button(
    entry_frame,
    text="Send",
    command=send_message,
    font=("Arial", 12, "bold"),
    bg="#00cc99",
    fg="black",
    width=10
)

send_button.grid(row=0, column=1)

# Enter Key Support
root.bind('<Return>', lambda event: send_message())

# Welcome Message
chat_area.config(state=tk.NORMAL)

chat_area.insert(
    tk.END,
    "Bot: Hello! Ask me anything about AI or Python.\n\n",
    "bot"
)

chat_area.config(state=tk.DISABLED)

# Run App
root.mainloop()