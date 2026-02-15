import pandas as pd
import seaborn as sns
from google.colab import files

Files = files.upload()

weather = pd.read_csv("weather_dataset.csv")
weather.info()
sns.barplot(weather["wind_speed"], weather["temperature"])
sns.displot(weather["temperature"])
sns.displot(weather["humidity"], rug = True)
sns.jointplot(weather["temperature"], weather["humidity"])
sns.jointplot(weather["temperature"], weather["humidity"], kind="hex")
sns.jointplot(weather["temperature"], weather["humidity"], kind="kde")
sns.pairplot(weather[["temperature", "humidity", "air_polution_index"]])
sns.stripplot(weather["weather_type"], weather["temperature"])
sns.stripplot(weather["weather_type"], weather["temperature"], jitter = True)
sns.swarmplot(weather["humidity"], weather["temperature"])
sns.boxenplot(weather["humidity"], weather["temperature"], hue = weather["weather_type"])
sns.barplot(weather["humidity"], weather["temperature"], hue = weather["weather_type"])
sns.countplot(weather["weather_type"])
sns.pointplot(weather["humidity"], weather["temperature"], hue = weather["weather_type"])
sns.implot(x = "humidity", y = "temperature", data = weather)
sns.implot(x = "humidity", y = "temperature", data = weather, hue = "weather_type")
