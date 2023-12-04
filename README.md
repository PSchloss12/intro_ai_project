# StoryBoard Project
Author: Patrick Schlosser

This project was designed for the Introduction to AI class at the University of Notre Dame. 
This model is designed to aid in the writing of stories especially childrens stories. It is not meant to answer questions but instead should be prompted with 5 words of a story. 
# Usage
The python functionality in this project is linked to a front end through Flask. Run the Flask app file app.py locally to start the project; it may take a minute to start up as it resolves data files. Then type at least five words into the text box into the textbox in the HTML webpage and git generate. You may need to increase the temperature to get longer responses. 
The "generate.py" file can also be run on its own in a terminal as long as you give it a new prompt each time.
# Known Erata
This model uses "\<eos\>" to mark the end of a generation sequence. Sometimes the model can get stuck on a line if the generating with a certain seed and temperature combination results in an immediate terminating symbol.
