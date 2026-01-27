import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

def isValidExtention(filePath):
	#Check if the extention is jpeg
	validExtention = ('.jpg', '.jpeg', '.JPG', '.JPEG')
	_, extention = os.path.splitext(filePath)

	if not extention in validExtention:
		return False

	return True

def getGraphData(dataPath, dirs, entries):
	#Parse every Directories and Count the number of Valid Files in the dataset
	for dir in dirs:
		dirPath = os.path.join(dataPath, dir)
		plantsNbr = sum(1 for entry in os.scandir(dirPath) if entry.is_file() and isValidExtention(entry.path))
		entries.append(plantsNbr)

	#Build the data struct used to create graphs
	data = {
		"Categories": dirs,
		"Nbr": entries
	}

	return data

def createGraphs(dirName, data):

	default_colors = plotly.colors.qualitative.Plotly

	#Setup the Graph Area
	fig = make_subplots(
		rows=1, cols=2,
		specs=[[{"type": "domain"}, {"type": "xy"}]]
	)

	#Create the Pie Graph
	fig.add_trace (
		go.Pie(labels=data['Categories'], values=data['Nbr'], marker=dict(colors=default_colors)), 
		row=1, col=1
	)

	#Create the Bar Grap
	fig.add_trace(
		go.Bar(x=data['Categories'], y=data['Nbr'], marker_color=default_colors),
		row=1, col=2
	)

	#Set Graph Title
	Title = dirName + " class distribution"
	fig.update_layout(title=Title)
	fig.show()

if __name__ == "__main__":

	#Basic Checks
	if len(sys.argv) <= 1 or len(sys.argv) > 2:
		print("Error: Expected Argument: 'Path to Directory'")
		exit(1)

	dataPath = sys.argv[1]
	dirName = dataPath.split("/")[-1]

	if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
		print("Error: Invalid Data Path Provided")
		exit(1)

	dirs = os.listdir(dataPath)
	entries = []

	data = getGraphData(dataPath, dirs, entries)

	createGraphs(dirName, data)

				
