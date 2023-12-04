// static/app.js
document.addEventListener('DOMContentLoaded', function () {
    const textarea = document.getElementById('storyboard');
    const generateBtn = document.getElementById('generateBtn');
    const maxLengthInput = document.getElementById('maxLength');
    const temperatureInput = document.getElementById('temperature');

    generateBtn.addEventListener('click', async function () {
        // Get the values from the input fields
        const maxLength = parseInt(maxLengthInput.value, 10);
        const temperature = parseFloat(temperatureInput.value);

        // Get the last 5 words from the textarea
        const text = textarea.value.trim();
        const lastFiveWords = text.split(/\s+/).slice(-5).join(' ');

        // Call the Flask backend to generate text with additional parameters
        const generatedText = await callFlaskBackend(lastFiveWords, maxLength, temperature);

        // Append the result to the textarea
        textarea.value = text.split(/\s+/).slice(0,-5).join(' ') + ' ' + generatedText;
    });

    async function callFlaskBackend(input, maxLength, temperature) {
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ input, maxLength, temperature }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            return result.generatedText;
        } catch (error) {
            console.error('Error:', error);
            return 'Error occurred while calling the backend';
        }
    }
});
