from typing import Type
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import get_prediction,label_encoder,get_severity

model=joblib.load('RTA model.sav')
st.set_page_config(page_title='Accident Severity Prediction App',page_icon='🚦',layout='wide')

#time=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
Day_of_week=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
Age_band_of_driver=['Under 18','18-30','31-50','Over 51','Unknown']
Sex_of_driver=['Male', 'Female', 'Unknown']
Educational_level=['Above high school', 'Junior high school', 'Elementary school','High school', 'Illiterate', 'Writing & reading','Unknown']
Vehicle_driver_relation=['Employee', 'Owner', 'Other','Unknown']
Driving_experience=['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence','Below 1yr', 'unknown']
Type_of_vehicle=['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)','Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry','Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen','Ridden horse', 'Bajaj', 'Turbo', 'Motorcycle','Special vehicle', 'Bicycle','Other']
Owner_of_vehicle=['Owner', 'Governmental', 'Organization', 'Other']
Area_accident_occurred=['Residential areas', 'Office areas', '  Recreational areas',' Industrial areas', 'Other', ' Church areas', '  Market areas','Unknown', 'Rural village areas', ' Outside rural areas',' Hospital areas', 'School areas','Rural village areasOffice areas', 'Recreational areas']
Lanes_or_Medians=['Two-way (divided with broken lines road marking)','Undivided Two way', 'Double carriageway (median)','One way', 'Two-way (divided with solid lines road marking)','Unknown','other']
Road_alignment=['Tangent road with flat terrain','Tangent road with mild grade and flat terrain', 'Escarpments','Tangent road with rolling terrain', 'Gentle horizontal curve','Tangent road with mountainous terrain and','Steep grade downward with mountainous terrain','Sharp reverse curve','Steep grade upward with mountainous terrain']
Types_of_junction=['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other','Unknown', 'T Shape', 'X Shape']
Road_surface_types=['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress','Gravel roads', 'Other']
Road_surface_conditions=['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
Light_conditions=['Daylight', 'Darkness - lights lit', 'Darkness - no lighting','Darkness - lights unlit']
Weather_conditions=['Normal', 'Raining', 'Raining and Windy', 'Cloudy','Windy', 'Snow', 'Unknown', 'Fog or mist','Other']
Type_of_collision=['Collision with roadside-parked vehicles','Vehicle with vehicle collision','Collision with roadside objects', 'Collision with animals','Other', 'Rollover', 'Fall from vehicles','Collision with pedestrians', 'With Train', 'Unknown']
#number_of_vehicles_involved 1-7
#number_of_casualties 1-8
Vehicle_movement=['Going straight', 'U-Turn', 'Moving Backward', 'Turnover','Waiting to go', 'Getting off', 'Reversing', 'Parked','Stopping', 'Overtaking', 'Other', 'Entering a junction','Unknown']
Casualty_class=['Driver or rider', 'Pedestrian', 'Passenger']
Sex_of_casualty=['Male', 'Female']
Age_band_of_casualty=['18-30', '31-50', 'Under 18', 'Over 51', '5']
Casualty_severity=[3.0, 2.0, 1.0]
Work_of_casuality=['Driver', 'Unemployed', 'Employee', 'Self-employed','Student','Other', 'Unknown']
Fitness_of_casuality=['Normal', 'Deaf', 'Other', 'Blind', 'NormalNormal']
Pedestrian_movement=['Not a Pedestrian', "Crossing from driver's nearside",'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle','Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle','In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)','Walking along in carriageway, back to traffic','Walking along in carriageway, facing traffic','In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle','Unknown or other']


st.markdown("<h1 style = 'text-align: center;'>Accident Severity Prediction Application 🤕</h1>", unsafe_allow_html=True)
video_html = """
            <center>
            <video controls width="750" autoplay="true" muted="true" loop="true">
<source 
            src="https://ak.picdn.net/shutterstock/videos/26865835/preview/stock-footage-animation-gifographic-accidents-injuries-danger-and-safety-caution-on-traffic-road-cause-by-cars.webm" 
            type="video/mp4" />
</video>
</center>

        """
st.markdown(video_html,unsafe_allow_html=True)


def main():
    with st.form('prediction_form'):
        st.subheader('Enter the values of the following features:')

        time=st.slider("Hour of accident",0,23,value=0,format='%d')
        day_of_week=st.selectbox("Select day of the week",options=Day_of_week)
        age_band_of_driver=st.selectbox("Select age band of the driver",options=Age_band_of_driver)
        sex_of_driver=st.selectbox("Select gender of the driver",options=Sex_of_driver)
        educational_level=st.selectbox("Select educational level of the driver",options=Educational_level)
        vehicle_driver_relation=st.selectbox("Select relation of the driver to the vehicle",options=Vehicle_driver_relation)
        driving_experience=st.selectbox("Select driving experience",options=Driving_experience)
        type_of_vehicle=st.selectbox("Select the type of vehicle",options=Type_of_vehicle)
        owner_of_vehicle=st.selectbox("Select owner of vehicle",options=Owner_of_vehicle)
        area_accident_occurred=st.selectbox("Select the area where accident occurred",options=Area_accident_occurred)
        lanes_or_Medians=st.selectbox("Select the type of lane or median",options=Lanes_or_Medians)
        road_alignment=st.selectbox("Select the type of road alignment",options=Road_alignment)
        types_of_junction=st.selectbox("Select the type of junction",options=Types_of_junction)
        road_surface_types=st.selectbox("Select the road surface type",options=Road_surface_types)
        road_surface_conditions=st.selectbox("Select the road surface conditions",options=Road_surface_conditions)
        light_conditions=st.selectbox("Select the lighting conditions",options=Light_conditions)
        weather_conditions=st.selectbox("Select the weather conditions",options=Weather_conditions)
        type_of_collision=st.selectbox("Select the type of collision occurred",options=Type_of_collision)
        number_of_vehicles_involved=st.slider("Select the number of vehicles involved",1,10,value=1,format='%d')
        number_of_casualties=st.slider("Select the number of casualties",1,10,value=1,format='%d')
        vehicle_movement=st.selectbox("Select the type of vehicle movement",options=Vehicle_movement)
        casualty_class=st.selectbox("Select the casualty class",options=Casualty_class)
        sex_of_casualty=st.selectbox("Select the gender of casualty",options=Sex_of_casualty)
        age_band_of_casualty=st.selectbox("Select the age band of casualty",options=Age_band_of_casualty)
        casualty_severity=st.selectbox("Select the casualty severity",options=Casualty_severity)
        work_of_casuality=st.selectbox("Select the work of casualty",options=Work_of_casuality)
        fitness_of_casuality=st.selectbox("Select the fitness of casualty",options=Fitness_of_casuality)
        pedestrian_movement=st.selectbox("Select the type of pedestrian movement",options=Pedestrian_movement)

        submit=st.form_submit_button("Predict")

    if submit:
        day_of_week=label_encoder(day_of_week,Day_of_week)
        age_band_of_driver=label_encoder(age_band_of_driver,Age_band_of_driver)
        sex_of_driver=label_encoder(sex_of_driver,Sex_of_driver)
        educational_level=label_encoder(educational_level,Educational_level)
        vehicle_driver_relation=label_encoder(vehicle_driver_relation,Vehicle_driver_relation)
        driving_experience=label_encoder(driving_experience,Driving_experience)
        type_of_vehicle=label_encoder(type_of_vehicle,Type_of_vehicle)
        owner_of_vehicle=label_encoder(owner_of_vehicle,Owner_of_vehicle)
        area_accident_occurred=label_encoder(area_accident_occurred,Area_accident_occurred)
        lanes_or_Medians=label_encoder(lanes_or_Medians,Lanes_or_Medians)
        road_alignment=label_encoder(road_alignment,Road_alignment)
        types_of_junction=label_encoder(types_of_junction,Types_of_junction)
        road_surface_types=label_encoder(road_surface_types,Road_surface_types)
        road_surface_conditions=label_encoder(road_surface_conditions,Road_surface_conditions)
        light_conditions=label_encoder(light_conditions,Light_conditions)
        weather_conditions=label_encoder(weather_conditions,Weather_conditions)
        type_of_collision=label_encoder(type_of_collision,Type_of_collision)
        #number_of_vehicles_involved 1-7
        #number_of_casualties 1-8
        vehicle_movement=label_encoder(vehicle_movement,Vehicle_movement)
        casualty_class=label_encoder(casualty_class,Casualty_class)
        sex_of_casualty=label_encoder(sex_of_casualty,Sex_of_casualty)
        age_band_of_casualty=label_encoder(age_band_of_casualty,Age_band_of_casualty)
        casualty_severity=label_encoder(casualty_severity,Casualty_severity)
        work_of_casuality=label_encoder(work_of_casuality,Work_of_casuality)
        fitness_of_casuality=label_encoder(fitness_of_casuality,Fitness_of_casuality)
        pedestrian_movement=label_encoder(pedestrian_movement,Pedestrian_movement)

        data=np.array([time,day_of_week,age_band_of_driver,sex_of_driver,educational_level,vehicle_driver_relation,
        driving_experience,type_of_vehicle,owner_of_vehicle,area_accident_occurred,lanes_or_Medians,road_alignment,
        types_of_junction,road_surface_types,road_surface_conditions,light_conditions,weather_conditions,type_of_collision,
        number_of_vehicles_involved,number_of_casualties,vehicle_movement,casualty_class,sex_of_casualty,age_band_of_casualty,
        casualty_severity,work_of_casuality,fitness_of_casuality,pedestrian_movement]).reshape(1,-1)

        pred=get_prediction(data=data,model=model)
        
        res=get_severity(pred[0])
        st.write(f"Predicted severity is: {res}")
        

if __name__=='__main__':
    main()