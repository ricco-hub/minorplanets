from asteroid_script import *

objects = ["bamberga", "ganymed", "pluto", "makemake"]
astsinfo = ["/home/r/rbond/sigurdkn/project/actpol/ephemerides/objects/Bamberga.npy", "/home/r/rbond/sigurdkn/project/actpol/ephemerides/objects/Ganymed.npy", 
"/home/r/rbond/sigurdkn/project/actpol/ephemerides/objects/Pluto.npy", "/home/r/rbond/sigurdkn/project/actpol/ephemerides/objects/Makemake.npy"]
arrs = ["pa4", "pa5", "pa6"]

k = 0 
while k<len(objects):
  #array pa4
  make_movie(astsinfo[k], objects[k], arrs[0], "f150")
  make_gallery(objects[k], arrs[0], "f150", directory = "asteroids/" + objects[k] + "/" + arrs[0] + "/f150/")  
  make_movie(astsinfo[k], objects[k], arrs[0], "f220")
  make_gallery(objects[k], arrs[0], "f220", directory = "asteroids/" + objects[k] + "/" + arrs[0] + "/f220/")  
  
  #array pa5
  make_movie(astsinfo[k], objects[k], arrs[1], "f090")
  make_gallery(objects[k], arrs[1], "f090", directory = "asteroids/" + objects[k] + "/" + arrs[1] + "/f090/")  
  make_movie(astsinfo[k], objects[k], arrs[1], "f150")
  make_gallery(objects[k], arrs[1], "f150", directory = "asteroids/" + objects[k] + "/" + arrs[1] + "/f150/")  
  
  #array pa6
  make_movie(astsinfo[k], objects[k], arrs[2], "f090")
  make_gallery(objects[k], arrs[2], "f090", directory = "asteroids/" + objects[k] + "/" + arrs[2] + "/f090/")  
  make_movie(astsinfo[k], objects[k], arrs[2], "f150")
  make_gallery(objects[k], arrs[2], "f150", directory = "asteroids/" objects[k] + "/" + arrs[2] + "/f150/")  
  
  k += 1              