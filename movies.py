from asteroid_script import *

objects = ["erminia", "egeria", "kalliope"]
astsinfo = "/home/r/rbond/sigurdkn/project/actpol/ephemerides/objects/" 
arrs = ["pa4", "pa5", "pa6"]

k = 0 
while k<len(objects):
  #array pa4
    #f150
  get_maps(astsinfo + objects[k].capitalize() + ".npy", objects[k], arrs[0], "f150")
  make_gallery(objects[k], arrs[0], "f150", directory = "test/" + objects[k] + "/" + arrs[0] + "/f150/")  
  flux_stack(objects[k], arrs[0], "f150", directory = "test/" + objects[k] + "/" + arrs[0] + "/f150/")  
    
    #f220
  get_maps(astsinfo + objects[k].capitalize() + ".npy", objects[k], arrs[0], "f220")
  make_gallery(objects[k], arrs[0], "f220", directory = "test/" + objects[k] + "/" + arrs[0] + "/f220/")   
  flux_stack(objects[k], arrs[0], "f220", directory = "test/" + objects[k] + "/" + arrs[0] + "/f220/")   
  
  #array pa5
    #f090
  get_maps(astsinfo + objects[k].capitalize() + ".npy", objects[k], arrs[1], "f090")
  make_gallery(objects[k], arrs[1], "f090", directory = "test/" + objects[k] + "/" + arrs[1] + "/f090/")
  flux_stack(objects[k], arrs[1], "f090", directory = "test/" + objects[k] + "/" + arrs[1] + "/f090/")  
    
    #f150
  get_maps(astsinfo + objects[k].capitalize() + ".npy", objects[k], arrs[1], "f150")
  make_gallery(objects[k], arrs[1], "f150", directory = "test/" + objects[k] + "/" + arrs[1] + "/f150/")
  flux_stack(objects[k], arrs[1], "f150", directory = "test/" + objects[k] + "/" + arrs[1] + "/f150/")    
  
  #array pa6
    #f090
  get_maps(astsinfo + objects[k].capitalize() + ".npy", objects[k], arrs[2], "f090")
  make_gallery(objects[k], arrs[2], "f090", directory = "test/" + objects[k] + "/" + arrs[2] + "/f090/")
  flux_stack(objects[k], arrs[2], "f090", directory = "test/" + objects[k] + "/" + arrs[2] + "/f090/")  
    
    #f150
  get_maps(astsinfo + objects[k].capitalize() + ".npy", objects[k], arrs[2], "f150")
  make_gallery(objects[k], arrs[2], "f150", directory = "test/" + objects[k] + "/" + arrs[2] + "/f150/")
  flux_stack(objects[k], arrs[2], "f150", directory = "test/" + objects[k] + "/" + arrs[2] + "/f150/")    
  
  k += 1              