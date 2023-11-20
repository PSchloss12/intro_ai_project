document.addEventListener('DOMContentLoaded', function () {
    const textarea = document.getElementById('storyboard');
    const generateBtn = document.getElementById('generateBtn');

    generateBtn.addEventListener('click', async function () {
        // Get the last 5 words from the textarea
        const text = textarea.value.trim();
        const lastFiveWords = text.split(/\s+/).slice(-5).join(' ');

        // Call the Python generate function in the background
        const generatedText = await callPythonBackend(lastFiveWords);

        // Append the result to the textarea
        textarea.value += '\n' + generatedText;

        // Optionally, you can clear the textarea after generating
        // textarea.value = '';
    });

    async function callPythonBackend(input) {
        return new Promise((resolve, reject) => {
            // Use Node.js child_process to call the Python script
            const { spawn } = require('child_process');
            const pythonProcess = spawn('python', ['generate.py', input]);

            let result = '';

            // Collect data from the Python script's stdout
            pythonProcess.stdout.on('data', (data) => {
                result += data.toString();
            });

            // Handle errors
            pythonProcess.stderr.on('data', (data) => {
                console.error(`Error: ${data}`);
                reject(data.toString());
            });

            // Handle the completion of the Python script
            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(result.trim());
                } else {
                    console.error(`Python script exited with code ${code}`);
                    reject(`Python script exited with code ${code}`);
                }
            });
        });
    }
});
