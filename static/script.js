document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const statusDiv = document.getElementById('status');
    const apiUrlInput = document.getElementById('api-url');
    const temperatureInput = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const maxLengthInput = document.getElementById('max-length');
    
    let messages = [
        { role: "assistant", content: "Hello! I'm an AI assistant. How can I help you today?" }
    ];
    
    // Update temperature value display
    temperatureInput.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
    });
    
    // Handle send button click
    sendBtn.addEventListener('click', sendMessage);
    
    // Handle Enter key to send message
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // Disable input and button
        userInput.disabled = true;
        sendBtn.disabled = true;
        
        // Display user message
        addMessageToChat('user', message);
        
        // Save message
        messages.push({ role: "user", content: message });
        
        // Clear input box
        userInput.value = '';
        
        // Show loading status
        statusDiv.innerHTML = 'Thinking <span class="loading"></span>';
        
        // Send request
        const apiUrl = apiUrlInput.value;
        const temperature = parseFloat(temperatureInput.value);
        const maxLength = parseInt(maxLengthInput.value);
        
        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                messages: messages,
                temperature: temperature,
                max_length: maxLength
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network request failed');
            }
            return response.json();
        })
        .then(data => {
            // Add model response
            const modelResponse = data.response;
            addMessageToChat('assistant', modelResponse);
            
            // Save message
            messages.push({ role: "assistant", content: modelResponse });
            
            // Clear status
            statusDiv.innerHTML = '';
        })
        .catch(error => {
            console.error('Error:', error);
            statusDiv.innerHTML = `Error: ${error.message}`;
        })
        .finally(() => {
            // Re-enable input and button
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        });
    }
    
    function addMessageToChat(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role === 'user' ? 'user-message' : 'model-message'}`;
        messageDiv.textContent = content;
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});