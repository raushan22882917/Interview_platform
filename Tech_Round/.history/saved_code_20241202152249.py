<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Interface</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='assets/css/styles.css') }}">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9f9f9;
        }

        .container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #fff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }

        .message-box {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            padding: 10px;
            border-radius: 10px;
        }

        .user-message {
            background: #e6f7ff;
            text-align: right;
        }

        .bot-message {
            background: #f0f0f0;
            text-align: left;
        }

        .input-section {
            display: flex;
            margin-top: 20px;
        }

        .input-section input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }

        .input-section button {
            background: #007bff;
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            margin-left: 10px;
            cursor: pointer;
        }

        .input-section button:hover {
            background: #0056b3;
        }

        .step-box {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #f9f9f9;
        }

        .step-number {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Chatbot Interface</h2>
        <div id="chatbox" class="message-box"></div>

        <div id="step-container" class="step-box" style="display: none;">
            <p class="step-number">Step: <span id="current-step"></span></p>
            <textarea id="step-input" rows="5" placeholder="Enter your response for this step..." style="width: 100%;"></textarea>
            <button id="submit-step">Submit Step</button>
        </div>

        <div class="input-section">
            <input type="text" id="question" placeholder="Ask your question...">
            <button id="submit-question">Submit</button>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function () {
            const steps = ["Example", "Solution", "Test Cases", "Code", "Validation"];
            let currentStepIndex = 0;

            function updateChatbox(message, type = "bot") {
                const chatbox = $('#chatbox');
                const msgClass = type === "user" ? "user-message" : "bot-message";
                chatbox.append(`<div class="message ${msgClass}">${message}</div>`);
                chatbox.scrollTop(chatbox[0].scrollHeight);
            }

            $('#submit-question').click(function () {
                const question = $('#question').val().trim();
                if (!question) {
                    alert("Please enter a question!");
                    return;
                }

                updateChatbox(question, "user");

                $.ajax({
                    url: '/submit_question',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ question }),
                    success: function (response) {
                        updateChatbox(response.message);
                        $('#question').val('');
                        $('#step-container').show();
                        $('#current-step').text(steps[currentStepIndex]);
                    },
                    error: function (error) {
                        console.log('Error:', error);
                        updateChatbox("Error submitting question. Please try again.");
                    }
                });
            });

            $('#submit-step').click(function () {
                const stepResponse = $('#step-input').val().trim();
                if (!stepResponse) {
                    alert("Please enter a response for the step!");
                    return;
                }

                updateChatbox(`Step ${steps[currentStepIndex]}: ${stepResponse}`, "user");

                $.ajax({
                    url: '/submit_step',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ step: steps[currentStepIndex], response: stepResponse }),
                    success: function (response) {
                        updateChatbox(response.message);
                        $('#step-input').val('');

                        currentStepIndex++;
                        if (currentStepIndex < steps.length) {
                            $('#current-step').text(steps[currentStepIndex]);
                        } else {
                            $('#step-container').hide();
                            updateChatbox("All steps completed. Submitting for evaluation...");
                            $.ajax({
                                url: '/evaluate',
                                type: 'POST',
                                success: function (response) {
                                    updateChatbox(response.message);
                                },
                                error: function (error) {
                                    console.log('Error:', error);
                                    updateChatbox("Error during evaluation. Please try again.");
                                }
                            });
                        }
                    },
                    error: function (error) {
                        console.log('Error:', error);
                        updateChatbox("Error submitting step. Please try again.");
                    }
                });
            });
        });
    </script>
</body>
</html>
