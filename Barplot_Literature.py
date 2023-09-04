import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv(r"C:\Users\PHYS3009\OneDrive - University of Waterloo\MSc - UW\Thesis\Chapter4\Figures\Barplot_Literature\scopus.csv")

# Counting the number of articles published each year
yearly_counts = data['Year'].value_counts().sort_index()

# Plotting the frequency of publications by year
plt.figure(figsize=(6,3))
yearly_counts.plot(kind='bar', color='grey', edgecolor='black', zorder=3)
#plt.title("Number of Publications by Year", fontsize=16)
plt.xlabel("Year", fontsize=12, fontname="Times New Roman")
plt.ylabel("Number of Publications", fontsize=12, fontname="Times New Roman")
plt.xticks(rotation=45, fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
plt.grid(axis='y', zorder=2)
plt.tight_layout()
plt.show()
