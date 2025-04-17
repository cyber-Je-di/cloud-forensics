
# OpenStack Log Anomaly Detection with Isolation Forest

## Step 1: Load and Parse the Logs

```python
import pandas as pd
import re

# Path to your OpenStack log file
log_file_path = 'openstack_abnormal.log'

# Load logs
with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    raw_logs = f.readlines()

# Function to parse log lines
def parse_log_line(line):
    pattern = r'^(\S+)\s+(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+)\s+(\d+)\s+(\w+)\s+(\S+)\s+.*?\s+(.*)$'
    match = re.match(pattern, line)
    if match:
        return {
            'log_source': match.group(1),
            'timestamp': match.group(2),
            'pid': match.group(3),
            'severity': match.group(4),
            'component': match.group(5),
            'message': match.group(6)
        }
    return None

parsed_logs = [parse_log_line(line) for line in raw_logs]
parsed_logs = [log for log in parsed_logs if log is not None]

df = pd.DataFrame(parsed_logs)
df.head()
```

## Step 2: Preprocess and Vectorize

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Fill missing messages (if any)
df['message'] = df['message'].fillna('')

# Convert message text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['message']).toarray()
```

## Step 3: Train Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Train anomaly detection model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Predict anomalies (-1 means anomaly, 1 means normal)
df['anomaly'] = model.predict(X)
df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})
```

## Step 4: View Results

```python
# See some anomalies
anomalies = df[df['anomaly'] == 'anomaly']
print(f"Total Anomalies Detected: {len(anomalies)}")
anomalies[['timestamp', 'severity', 'component', 'message']].head(10)
```

## Optional: Save Anomaly Report

```python
# Save to CSV
anomalies.to_csv("anomaly_report.csv", index=False)
print("Anomaly report saved as 'anomaly_report.csv'")
```
