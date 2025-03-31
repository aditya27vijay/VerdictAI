import React, { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8001/chat"; // Change this when deployed

function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!query.trim()) return;

    const newMessages = [...messages, { sender: "user", text: query }];
    setMessages(newMessages);
    setQuery("");
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      const botReply = data.response || "Error fetching response.";
      setMessages([...newMessages, { sender: "bot", text: botReply }]);
    } catch (error) {
      setMessages([...newMessages, { sender: "bot", text: "Error: Try again later." }]);
    }

    setLoading(false);
  };

  return (
    <div className="chat-container">
      <h1>🧑‍⚖️ Legal Chatbot</h1>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
          {msg.sender === "bot" ? (
            <p dangerouslySetInnerHTML={{ __html: msg.text.replace(/\n/g, "<br>") }}></p>
          ) : (
            <p>{msg.text}</p>
          )}
        </div>
        ))}
        {loading && <div className="message bot">Typing...</div>}
      </div>
  

      <div className="input-area">
        <input
          type="text"
          placeholder="Ask a legal question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage} disabled={loading}>Send</button>
      </div>
    </div>
  );
}

export default App;
