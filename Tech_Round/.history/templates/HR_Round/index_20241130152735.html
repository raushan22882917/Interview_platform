<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Interview Bot</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div id="chat-container">
        <div id="chat-box"></div>
        <div class="input-container">
            <input type="text" id="company-name" placeholder="Enter company name...">
            <button onclick="startInterview()">Start Interview</button>
        </div>

        <div class="input-container" id="answer-container" style="display: none;">
            <input type="text" id="user-answer" placeholder="Your answer...">
            <button onclick="submitAnswer()">Submit</button>
        </div>
    </div>

    <script>
        let companyName = "";
        const questions = [
            "Can you tell me about yourself?",
            "Why do you want to work at {company_name}?",
            "What are your greatest strengths?",
            "Where do you see yourself in five years?",
            "Why should we hire you over other candidates?",
            "Describe a challenging situation you faced and how you resolved it.",
            "What motivates you to perform well at work?",
            "How do you handle criticism or feedback from your manager?",
            "Can you share an example of how you worked effectively in a team?",
            "What is your approach to time management and meeting deadlines?"
        ];

        let currentQuestionIndex = 0;
        let answeredQuestions = 0; // Counter for answered questions
        let formattedQuestions = [];

        // Start the interview by asking the first question
        function startInterview() {
            companyName = document.getElementById("company-name").value.trim();
            if (!companyName) {
                alert("Please enter a company name.");
                return;
            }

            // Replace placeholder with company name in questions
            formattedQuestions = questions.map(question => question.replace("{company_name}", companyName));

            // Display the first question
            askQuestion();
            document.getElementById("answer-container").style.display = "block";
            document.getElementById("company-name").disabled = true;
            document.querySelector("button").disabled = true;  // Disable start button after starting the interview
        }

        // Display question and speak it
        function askQuestion() {
            if (currentQuestionIndex < formattedQuestions.length) {
                const question = formattedQuestions[currentQuestionIndex];
                appendMessage("Bot", question);
                speakQuestion(question);
            }
        }

        // Append message to chat
        function appendMessage(sender, message) {
            const chatBox = document.getElementById("chat-box");
            const messageDiv = document.createElement("div");
            messageDiv.classList.add(sender);
            messageDiv.textContent = message;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
        }

        // Speak the question using the Web Speech API
        function speakQuestion(question) {
            const speech = new SpeechSynthesisUtterance(question);
            window.speechSynthesis.speak(speech);
        }

        // Submit answer and send it to the backend
        function submitAnswer() {
            const userAnswer = document.getElementById("user-answer").value;
            if (userAnswer.trim() === "") {
                alert("Please provide an answer!");
                return;
            }

            // Send the answer to the backend
            fetch('/submit_answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: new URLSearchParams({
                    company_name: companyName,
                    question_index: currentQuestionIndex,
                    user_answer: userAnswer
                })
            })
            .then(response => response.json())
            .then(data => {
                appendMessage("User", userAnswer);
                appendMessage("Bot", `Feedback: ${data.feedback}`);
                appendMessage("Bot", `Score: ${data.score}/5`);

                // Increment the question counter
                answeredQuestions++;

                // After answering the current question, move to the next
                currentQuestionIndex++;
                document.getElementById("user-answer").value = ""; // Clear input field

                // If there are more questions, ask the next question
                if (currentQuestionIndex < formattedQuestions.length) {
                    setTimeout(askQuestion, 2000); // Wait for 2 seconds before asking the next question
                } else {
                    appendMessage("Bot", "You have completed the interview!");
                }
            })
            .catch(error => {
                console.error("Error:", error);
            });
        }
    </script>
</body>
</html>
