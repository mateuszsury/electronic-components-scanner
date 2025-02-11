# Electronic Component Datasheet Finder

An AI-powered web server application for recognizing electronic components and retrieving their datasheets—whether by processing an image of the component or by manual input of its designation.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

This project is a complete application featuring both backend and frontend components. It leverages artificial intelligence to recognize electronic components from images and/or manual inputs. Once the component is identified, the application scrapes data from popular online databases to fetch the relevant datasheet, providing users with accurate and up-to-date information.

---

## Features

- **AI-Powered Recognition:**  
  Utilizes AI algorithms to accurately identify electronic components from uploaded images.

- **Manual Search Option:**  
  Allows users to directly enter component designations to search for datasheets.

- **Data Scraping:**  
  Automatically scrapes and compiles datasheet information from the most popular electronic component databases.

- **User-Friendly Interface:**  
  Provides a simple and intuitive web interface for effortless interaction.

- **Complete Integration:**  
  Combines both frontend and backend systems to deliver a seamless user experience.

---

## Installation

Follow these steps to set up the application locally:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mateuszsury/electronic-components-scanner.git
   ```

2. **Extract the Repository:**

   If you have downloaded the repository as a ZIP file, extract it to your desired directory.

3. **Install Required Libraries:**

   Ensure you have Python installed, then install the necessary dependencies by running:

   ```bash
   pip install Flask difflib bs4 tqdm
   ```

4. **Run the Application:**

   Navigate to the project directory and start the server with:

   ```bash
   python server.py
   ```

---

## Usage

Once the server is running, you can access the application via your web browser:

### Open Your Browser:

Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) (or the URL provided in your terminal).

### Using the Interface:

- **Image Recognition:**  
  Upload an image of the electronic component. The AI module will process the image, identify the component, and then search for the corresponding datasheet.

- **Manual Search:**  
  Enter the component designation into the search bar to directly retrieve its datasheet.

---

## Dependencies

The project requires the following Python libraries:

- **Flask:** A micro web framework for Python.
- **difflib:** A module that provides classes and functions for comparing sequences.
- **Beautiful Soup (bs4):** A library for parsing HTML and XML documents.
- **tqdm:** A fast, extensible progress bar for loops and other iterative processes.

Make sure these libraries are installed before running the application.

---

## License

This project is licensed under the MIT License.
