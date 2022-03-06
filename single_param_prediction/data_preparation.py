import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_json("../data/marsWeather_till_17_1_2022.json")

df['terrestrial_date']=pd.to_datetime(df['terrestrial_date'])

new_df=pd.DataFrame(list(df['max_temp']),index=df.terrestrial_date,columns=['Maxtemperature'])


plt.figure(figsize=(20,10))
plt.plot(new_df)
# plt.show()

df.style.background_gradient(cmap ='viridis').set_properties(**{'font-size': '20px'})