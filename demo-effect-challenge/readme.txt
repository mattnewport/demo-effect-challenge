This is a work in progress. I didn't have time to convert much of the code to x64 assembly. A couple of the core
functions are implemented in assembly and I converted most of the heavy lifting code to SSE intrinsics but didn't have
time to assemblify everything. I also didn't get round to bringing everything over into the challenge framework - this
solution is still using my own framework / testbed.

My effect is a 2D fire simulation, essentially implementing the technique described in this paper: 
http://www.nik.no/2006/Gundersen2.pdf . The core of the simulation is based around Jos Stam's stable fluid solver (the 
same fluid solver Soleil implemented for his entry). There are three fluid density fields representing heat, fuel 
gas concentration and exhaust gas concentration. In cells where the heat exceeds a combustion temperature, combustion 
occurs which uses up fuel gas and creates exhaust gas and heat and generates buoyancy and expansion forces to apply to 
the velocity field. 

Fuel is added to the simulation each frame and an image data file is used for the 'fuel map'. Heat and fuel can also 
be added with the mouse, and the mouse can be used to mess around with the velocity field.

One of the reasons implementing this took longer than anticipated is that there are a lot of parameters to tweak to 
adjust the behaviour and look of the fire. I didn't really end up with something I was totally happy with and I started 
to build a C# app to tweak the parameters on the fly. I didn't complete that either though! There's also a bunch of other
code in app.cpp implementing Perlin noise and a simpler fire effect which I was playing around with at various stages. 
The code is pretty messy and in need of some cleanup.