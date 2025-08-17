# Twitter Political Sentiment Analysis

A Python-based sentiment analysis tool for analyzing political party opinions on Twitter using natural language processing and machine learning techniques.

## 🚧 Project Status
**Note: This project is not complete.**

## Features

- **Spanish Language Processing**: Uses spaCy's Spanish NLP model for tokenization and lemmatization
- **Political Sentiment Analysis**: Analyzes tweets for positive, negative, neutral, or no sentiment
- **Party-specific Analysis**: Processes multiple political parties' tweet data
- **Temporal Analysis**: Tracks sentiment trends over time
- **Visualization**: Generates charts and graphs for data visualization
- **Victory Probability**: Calculates simplified victory probabilities based on sentiment
- **Comprehensive Reports**: Creates detailed analysis reports

## Project Structure

```
├── README.md
├── requirements.txt           # Python dependencies
├── config.py                 # Configuration file with party names and months
├── main_version1.py          # Simplified version of the analysis
├── main_version2.py          # Extended version with broader dictionary
├── diccionario_positivo.txt  # Custom positive words dictionary (optional)
├── diccionario_negativo.txt  # Custom negative words dictionary (optional)
├── graficas/                 # Generated charts and visualizations
├── reportes/                 # Analysis reports
└── [month_folders]/          # CSV files organized by month and party
    └── [party_folders]/
        └── *.csv             # Tweet data files
```

## Installation

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Additional Dependencies
```bash
# Download Spanish spaCy model
python -m spacy download es_core_news_sm

# Download NLTK stopwords (run in Python)
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### Run the Analysis
```bash
# Simplified version
python main_version1.py

# Extended version with broader dictionary
python main_version2.py
```

## Key Components

### Preprocessing Features
- **Tweet Cleaning**: Removes URLs, mentions, hashtags, numbers, and special characters
- **Lemmatization**: Uses spaCy for Spanish text lemmatization
- **Stopword Removal**: Custom Spanish stopwords including political and social media terms
- **Retweet Processing**: Extracts original usernames from retweets

### Sentiment Analysis
- **Custom Dictionaries**: Predefined positive and negative word sets for political context
- **Sentiment Classification**: Categorizes tweets as positive, negative, neutral, or no sentiment
- **Word Frequency Analysis**: Tracks most common words and sentiment-specific terms

### Visualization & Reports
- **Sentiment Distribution**: Pie charts showing sentiment breakdowns
- **Word Frequency**: Bar charts of most frequent words
- **Temporal Trends**: Time-based sentiment analysis
- **Victory Probabilities**: Comparative analysis between parties
- **Detailed Reports**: Comprehensive text reports with statistics

## Configuration

The project uses a `config.py` file to define:
- `PARTIDOS`: List of political parties to analyze
- `MESES`: Months/directories to process
- `ARCHIVOS`: CSV file patterns to include

## Data Structure

Expected CSV file structure:
- First column: Tweet text
- Files organized in directories by month and political party
- Date information extracted from directory names (DDMMYY format)

## Output Files

### Generated Graphics
- `[party]_sentimientos_pie.png`: Sentiment distribution pie chart
- `[party]_palabras_frecuentes.png`: Most frequent words bar chart
- `[party]_palabras_pos_neg.png`: Positive/negative words comparison
- `[party]_tendencia_temporal.png`: Temporal sentiment trends
- `probabilidades_victoria.png`: Victory probability comparison

### Analysis Files
- `palabrasPositivos.txt`: Positive words found per user
- `palabrasNegativos.txt`: Negative words found per user
- `diccionario_palabras.txt`: Global word dictionary with sentiment labels
- `reporte_global.txt`: Comprehensive analysis report

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **matplotlib/seaborn**: Data visualization
- **spaCy**: Spanish NLP processing
- **nltk**: Natural language toolkit for stopwords
- **numpy**: Numerical computing

### Language Support
- Primarily designed for Spanish tweets
- Custom Spanish stopwords and political terminology
- Mexican political context considerations

## Limitations

⚠️ **Important Notes:**
- This is a simplified sentiment analysis approach
- Results should be interpreted as social media opinion trends, not precise electoral predictions
- Does not account for tweet reach, sample representativeness, or actual voting intentions
- Currently under development - features may be incomplete

## Future Enhancements

- Machine learning-based sentiment classification
- Real-time tweet processing
- Enhanced temporal analysis
- Multi-language support
- API integration for live data

## License

This project is open-source and available for educational and research purposes.