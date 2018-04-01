
# ChargedParticlesCS - README
* Copyright (c) 2012-2018  Thomas Mueller
* Email: tauzero7@gmail.com
* Article:  
    T. Mueller and J. Frauendiener,  
    __Charged particles constrained to a curved surface__,  
    European Journal of Physics __34__(1), 147-160 (2013)  
    [http://stacks.iop.org/0143-0807/34/i=1/a=147]()
* Preprint:  
    [https://arxiv.org/abs/1209.4184](arXiv:1209.4184v2)


---

## Prerequisite

* CUDA and graphics drivers

    Please note that this application works only on NVidia graphics boards
    that support the 'Compute Unified Device Architecture' (CUDA) and
    compute capability 1.1 or higher.  
    
    To check if your board should work have a look at here:
    [http://developer.nvidia.com/cuda/cuda-gpus]()
    or search for 'nvidia compute capability'.  
    
    You also need a more recent driver... 
    [http://www.nvidia.com/Download/index.aspx]()
    (Test the software with your current driver before installing 
    a new one. Maybe, it works already with your current driver.)  
    
    ... and the CUDA toolkit  
    [http://developer.nvidia.com/cuda/cuda-downloads]()


* Freeglut

    On Linux systems, the open source version of the OpenGL Utility 
    Toolkit can be installed from the package manager. Otherwise, you
    will find the sources here: [http://freeglut.sourceforge.net/]()
    Windows users can use Martin Payne's Windows binaries:
    [http://www.transmissionzero.co.uk/software/freeglut-devel/]()


* Font rendering

    You also need freetype2 installed for text output within
    the OpenGL context: [http://sourceforge.net/projects/freetype/]()  
    
    A Windows version can be found here (Binaries package): 
    [http://gnuwin32.sourceforge.net/packages/freetype.htm]()
    
    Maybe, you also have to download the Dependencies package and
    copy the bin/zlib1.dll file into the release / debug folder.
    

--- 

## Installation - LINUX

1. Unzip the ChaPaCS.zip archive and change to the directory ChaPaCS.

2. Open 'Makefile' in a standard text editor (e.g. gedit,kate)
    and adjust the following directory paths  
    - CUDA_DIR     = /usr/local/cuda  
    - FREETYPE_DIR = /usr/include/freetype2  
    depending on where these directories reside on your system.

3. Depending on the compute capability of your graphics board
    adjust the variables  
    - COMPUTE_CAPABILITY = compute_11  (compute_13, compute_20,...)
    - COMPUTE_PRECISION  = float       (double)

4. As CUDA only works with gcc compilers with a version <= 4.4,
    you might have to install an appropriate gcc version. In that
    case, you have to adjust the compilers  
    - CC  = gcc   (/usr/local/gcc/4.3/bin/gcc)  
    - CPP = g++   (/usr/local/gcc/4.3/bin/g++)  
    The CUDA compiler 'nvcc' must know the compiler binary directory,
    here: (/usr/local/gcc/4.3/bin)  

5. Depending on the memory capacity of your graphics board adjust  
    - FBO_TEXTURE_SIZE=512  (1024,2048,...)

6. make -f Makefile


    Each time you change one of the program files or the 'Makefile' you 
    have to compile the project. Sometimes, you have to clean all files
    before compiling the project:  
        make -f Makefile clean  
        make -f Makefile




## Installation - WINDOWS

1. Unzip the ChaPaCS.zip archive and change to the directory ChaPaCS.

2. Copy the freeglut and freetype packages in the ChaPaCS directory.

3. Open the solution vs2010/ChargedParticlesVS.sln with VisualStudio2010.

4. Go to "View -> Property manager" and expand "ChargedParticlesVS". 
    Then, double-click on the "cuda" property sheet. Select "User Macros"
    and adjust the macro variables  
    - FREEGLUT_DIR  
    - FREETYPE_DIR  
    - PRECISION_IN_USE    USE_DOUBLE           or let be empty  
    - COMPUTE_CAPABILITY  compute_20,sm_20     or compute_11,sm_11  or ...  
    - FBO_TEX_SIZE        512                  or 1024 or 2048 or ...  
    
    Afterwards, press Ctrl+S to save the property sheet.

5. Select  "Build -> Build Solution".	

6. Copy freeglut/bin/freeglut.dll and freetype/bin/freetype6.dll in the 
    vs2010/debug or vs2010/release directory, respectively.
    Possibly, you also have to copy bin/zlib1.dll from the freetype 
    dependency package also in the debug or release directory.

7. To start the program, open a "command prompt" and change to the 
    directory ChaPaCS. Then, start with

    vs2010\Debug\ChargedParticlesCS

    Please note that you start the program always from this directory,
    otherwise it will not work.


---

## Examples

To show command line parameters, start the program with  
    Linux:    ./ChargedParticlesCS -h  
    Windows:  vs2010\Debug\ChargedParticlesCS -h  


### Single sphere
  
   Set "USE_SCENE" to 0 in the main.inl file and recompile the code.
   Start the program with
   
   ./ChargedParticlesCS -log
   
   The command line argument "-log" causes the program to write
   the log file "output/log_energy.dat" that stores the kinetic and 
   field energy for each time step.
   
   Start the simulation by pressing 'p'. Press 'h' for help.

   To visualize the energies with respect to time, you can use for
   example "gnuplot". Start gnuplot and use
   
     plot "log_energy.dat" u 1:2 w l
     
   to plot the kinetic energy versus time.



---

## Add a new surface type

The following steps are necessary to add a new surface type. Replace
'mySurface' by an appropriate name. You might consult 'wxMaxima' to help
you in calculating the Christoffel symbols.

* Append a new enum to  "defs.h" -> e_surface_type:  
     e_surface_mySurface

* Add a new surface definition file in the 'cuda-directory':  
     cs_mySurface.inl  
   and define the following functions:  
    - mySurface_calc_f  
    - mySurface_calc_normal  
    - mySurface_calc_df  
    - mySurface_calc_metric  
    - mySurface_calc_invMetric  
    - mySurface_calc_chris  

* Open "src/GLShader.cpp" and append 'SURF_TYPE_MY_SHADER'
    in the 'readShaderFromFile' method. The number MUST be equal to 
    the e_surface_mySurface number in "defs.h" !

* Open "cuda/curvedSurfaceCode.cu" and append an include statement to
    "cs_mySurface.inl" and append 'SURF_TYPE_MY_SHADER' to the definitions. 
    Go through the complete code, add a switch-case with 
    SURF_TYPE_MY_SHADER, and insert the corresponding functions.

* Open "shaders/curvedSurface.vert" and append a function 'mySurface'
    similar to the other surface functions. Add a switch-case with
    'SURF_TYPE_MY_SHADER'.

* Open "shaders/particleMapping.geom" and append a function 'mySurface'
    where the particle splats are scaled by the inverse metric of the
    surface.

---