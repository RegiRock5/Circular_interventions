import mario
from mario import slicer, parse_exiobase_3,parse_from_txt, hybrid_sut_exiobase
# %%
regionlock = False #nl and rest of the world
regionlock2 = False # bigger regions in the world
sectorlock = False # arbitary sector combinations
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Output/" 

#%% input path of the IOT 
iot_path = r"C:/Industrial_ecology/Thesis/IOT_2011_ixi"
save_path = r'C:/Industrial_ecology/Thesis/Circularinterventions/Code'
world_IOT = parse_exiobase_3(path=iot_path, version='3.8.1')
#world_IOT.get_shock_excel(path=save_path)

#%%
if regionlock == True: 
    world_IOT.aggregate(r'Dutch_agg.xlsx', ignore_nan= True, levels = "Region")
    RegionI= "NL"
    world_IOT.get_index("Region")
    
if regionlock2 == True: 
    world_IOT.aggregate(r'Bigregion.xlsx', ignore_nan= True, levels = "Region")
    RegionI= "NL"
    world_IOT.get_index("Region")
    
#%% Uncomment if in need of shockexcel 
#world_IOT.get_aggregation_excel(path=r'sector.xlsx', levels= "Sector")

#%% Implement Aluminiums shocks (shock 1)
world_IOT.shock_calc(io=r'shocks_al.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                Y= True,
                e= True,
                notes=['Implement Aluminium shocks']
              )
#%%Implement Steel shocks(shock 2)
world_IOT.shock_calc(io=r'shocks_st.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                Y = True,
                e = True,
                notes=['Implement Steel shocks']
              )
#%%Implement both aluminium and steel shocks (shock 3)
world_IOT.shock_calc(io=r'shocks_full.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                Y= True,
                e= True,
                notes=['Implement Aluminium shocks']
              )
#%%Implement sectoral aggregation to reduce sector 
if sectorlock == True: 
    world_IOT.aggregate(r'sector.xlsx', ignore_nan= True, levels = "Sector")
    world_IOT.get_index("Sector")


#%% Create variables which will be used in the enviromental footprint and territorial analysis 
A_Al_adjusted = world_IOT['shock 1']['z']
Y_Al_adjusted = world_IOT['shock 1']['Y']
e_Al_adjusted =world_IOT['shock 1']['E']

A_St_adjusted = world_IOT['shock 2']['z']
Y_St_adjusted = world_IOT['shock 2']['Y']
e_St_adjusted =world_IOT['shock 2']['E']

A_full_adjusted = world_IOT['shock 3']['z']
Y_full_adjusted = world_IOT['shock 3']['Y']
e_full_adjusted =world_IOT['shock 3']['E']

#%%Export all variables to csv files
A_Al_adjusted.to_csv(f'{outputpath}A_Al_adjusted.csv', index=True)  
Y_Al_adjusted.to_csv(f'{outputpath}Y_Al_adjusted.csv', index=True)
e_Al_adjusted.to_csv(f'{outputpath}F_Al_adjusted.csv', index=True)

A_St_adjusted.to_csv(f'{outputpath}A_St_adjusted.csv', index=True)  
Y_St_adjusted.to_csv(f'{outputpath}Y_St_adjusted.csv', index=True)
e_St_adjusted.to_csv(f'{outputpath}F_St_adjusted.csv', index=True)

A_full_adjusted.to_csv(f'{outputpath}A_full_adjusted.csv', index=True)  
Y_full_adjusted.to_csv(f'{outputpath}Y_full_adjusted.csv', index=True)  
e_full_adjusted.to_csv(f'{outputpath}F_full_adjusted.csv', index=True)


