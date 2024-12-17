let mediaRecorder;
let recordedChunks = [];
let webcamStream;

async function startRecording() {
    const displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });

    // Combine display and webcam streams into one
    const combinedStream = new MediaStream([...displayStream.getTracks(), ...webcamStream.getTracks()]);

    mediaRecorder = new MediaRecorder(combinedStream);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);

        const preview = document.getElementById('preview');
        preview.src = url;

        const downloadLink = document.createElement('a');
        downloadLink.href = url;
        downloadLink.download = 'screen_recording_with_webcam.webm';
        document.body.appendChild(downloadLink);

        // Automatically trigger the download
        downloadLink.click();

        // Cleanup
        URL.revokeObjectURL(url);
        recordedChunks = [];
    };

    // Display webcam feed in the small video element
    const webcamPreview = document.getElementById('webcamPreview');
    webcamPreview.srcObject = webcamStream;

    mediaRecorder.start();
    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
}

function stopRecording() {
    mediaRecorder.stop();
    webcamStream.getTracks().forEach(track => track.stop()); // Stop the webcam stream
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
}


document.addEventListener('DOMContentLoaded', () => {
    const chatbotContainer = document.getElementById('chatbot-container');
    const chatbotClose = document.getElementById('chatbot-close');
    const chatbotMessages = document.getElementById('chatbot-messages');
    const chatbotText = document.getElementById('chatbot-text');
    const chatbotSend = document.getElementById('chatbot-send');
    const chatbotIcon = document.getElementById('chatbot-icon');

    // Initialize chatbot to be hidden
    chatbotContainer.style.display = 'none';

    // Toggle chatbot visibility
    chatbotIcon.addEventListener('click', () => {
        if (chatbotContainer.style.display === 'none') {
            chatbotContainer.style.display = 'flex';
        } else {
            chatbotContainer.style.display = 'none';
        }
    });

    // Close chatbot
    chatbotClose.addEventListener('click', () => {
        chatbotContainer.style.display = 'none';
    });

    // Send message
    chatbotSend.addEventListener('click', () => {
        const message = chatbotText.value;
        if (message.trim() !== '') {
            displayMessage('user', message);
            chatbotText.value = '';

            // Simulate a chatbot response
            setTimeout(() => {
                const response = getChatbotResponse(message);
                displayMessage('bot', response);
            }, 1000);
        }
    });

    function displayMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.className = sender;
        messageElement.textContent = message;
        chatbotMessages.appendChild(messageElement);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    function getChatbotResponse(userInput) {
        // Normalize the user input
        const normalizedInput = userInput.toLowerCase().trim();

        // Define the chatbot logic
        if (normalizedInput === 'how to run code') {
            return 'To run code, use the following method: `output = your_method_name(input_values)`.';
        } else if (normalizedInput === 'how to run editor') {
            return 'To run the editor, follow these steps: [Insert steps here].';
        } else {
            return 'I\'m sorry, I didn\'t understand that. Please choose "how to run code" or "how to run editor".';
        }
    }
});



function startRecording() {
    // Request access to the user's webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
            // Show the webcam preview
            document.getElementById('webcamPreview').srcObject = stream;

            // Create a new MediaRecorder for the stream
            mediaRecorder = new MediaRecorder(stream);

            // Store the recorded data chunks
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };

            // Start recording
            mediaRecorder.start();

            // Toggle button visibility
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'block';
        })
        .catch(error => {
            console.error('Error accessing media devices.', error);
        });
}

function stopRecording() {
    // Stop the MediaRecorder
    mediaRecorder.stop();

    // Combine the recorded chunks into a single Blob
    const blob = new Blob(recordedChunks, {
        type: 'video/webm'
    });

    // Create a URL for the recorded video
    const videoURL = URL.createObjectURL(blob);

    // Set the video preview source to the recorded video
    const preview = document.getElementById('preview');
    preview.src = videoURL;
    preview.controls = true;

    // Stop all video streams to release the webcam
    const tracks = document.getElementById('webcamPreview').srcObject.getTracks();
    tracks.forEach(track => track.stop());

    // Toggle button visibility
    document.getElementById('startBtn').style.display = 'block';
    document.getElementById('stopBtn').style.display = 'none';
}
