# SigmaNEAT

## Acknowledgements
An implementation of Hyper NEAT Algorithm with some practices.
This project started off by using the famous [MultiNEAT](http://multineat.com) project.

Python mine sweeper implementation thanks to Mohd-Akram's simple implementation [here](https://gist.github.com/mohd-akram/3057736). Of course it has been edited a lot to incorporate the AI needs.

## How to use
0. I use Visual Studio so its good if you do that too
0. You need to build the C++ *HyperNEAT* project. To do that:
   1. Download and build [boost](https://www.boost.org/). I currently use 1.69 but these guys are excelent programmers and will keep backward compatibility.
   2. Add boost include and lib dirs to *HyperNEAT* project configurations.
   3. You should be able to build and see **.pyd** file under *./release* folder.
0. You can now start any *main.py* file in any of the python projects and see the resuls.