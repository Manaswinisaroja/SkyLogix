from vpython import *

scene = canvas(title="Standalone Quadcopter Demo",
               width=900, height=600,
               background=vector(0.6,0.85,1))

scene.center = vector(0,5,0)

# ----------------------------
# DRONE CREATION FUNCTION
# ----------------------------

def create_drone(pos):
    body = box(pos=pos, size=vector(3,1,3), color=color.white)
    
    arms=[]; motors=[]; props=[]
    dirs=[vector(2,0,2), vector(2,0,-2),
          vector(-2,0,2), vector(-2,0,-2)]
    
    for d in dirs:
        arms.append(cylinder(pos=pos, axis=d, radius=0.15))
        mpos = pos + d
        motors.append(cylinder(pos=mpos, axis=vector(0,0.5,0), radius=0.3))
        props.append(box(pos=mpos+vector(0,0.6,0),
                         size=vector(1.6,0.1,0.3),
                         color=color.black))
    
    package = box(pos=pos-vector(0,1.5,0),
                  size=vector(1,1,1),
                  color=color.orange)
    
    return {"pos":pos,"body":body,"arms":arms,
            "motors":motors,"props":props,
            "package":package}

# ----------------------------
# CREATE DRONE
# ----------------------------

drone = create_drone(vector(0,10,0))

nav_light = sphere(pos=drone["body"].pos+vector(0,1.2,0),
                   radius=0.3,
                   color=color.red,
                   emissive=True)

# ----------------------------
# SIMPLE ANIMATION LOOP
# ----------------------------

t = 0
while True:
    rate(60)
    
    # Rotate propellers
    for p in drone["props"]:
        p.rotate(angle=1.5, axis=vector(0,1,0), origin=p.pos)
    
    # Make drone hover up and down smoothly
    hover = 0.05 * sin(t)
    t += 0.1
    
    move = vector(0, hover, 0)
    
    drone["pos"] += move
    
    # Move all parts together
    parts = [drone["body"], drone["package"]] + drone["arms"] + drone["motors"] + drone["props"]
    for part in parts:
        part.pos += move
    
    # Move navigation light
    nav_light.pos = drone["body"].pos + vector(0,1.2,0)
