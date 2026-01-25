# Moog Ladder Filters

This project contains different digital implementations of the classic 4-pole, 24 dB/octave analog filter introduced in 1965. The filter is well-regarded to add a nice character to any sound source, either synthesized or acoustic. The ladder structure consists of four one-pole filters and a global negative feedback loop.

The C++11 filter classes do not rely on external libraries and can be used with little to no modification in other DSP projects; the idea of this project is that you choose which variant suits your project best and then copy the parts that you need, rather than included as a monolithic DSP library. Every filter has been modified from its original implementation for code clarity and/or runtime performance. The project includes a test app that will load a sample wav file through each of the implemented filter variants. 

# Filter Tuning & A Word of Warning
Each model is unique. The newest is from 2025 while the oldest dates back over 20 years. Some try to remain true to their analog counterpart, where others are more approximate. The filters have not been rigorously verified for all combinations of cutoff, resonance, and sampling rate. Some are purposely built to self-oscillate, but beware the occasional blow-up with parameters that exceed some undiscovered value. 

# Models & Licenses

“Closed-Source Friendly” indicates if the individual license permits redistribution in a closed-source product (like a VST plugin). Filtered output audio is fair game for any kind of sample library or music production, commercial or otherwise. In the case of copyright-only code, it is possible to contact the original author to request an explicit license.

Implementation | License | Original Source | Closed-Source Friendly
------------- | ------------- | ----------------- | -----------------
Simplified | Custom | DAFX | No
Huovilainen  | LGPLv3 | CSound | If dynamically linked
Stilson | Unlicense | Moog~ by D. Lowenfels | Yes
Microtracker | Unlicense | Magnus Jonsson | Yes
Krajeski | Unlicense | Aaron Krajeski | Yes
MusicDSP | Suggested CC-BY-SA | MusicDSP.org | Yes
Oberheim | Custom | Will Pirkle | Yes
Improved | ISC | Via Author | Yes
RKSimulation | BSD | Bob~ by Miller Puckette | Yes
Hyperion | Unlicense | Via Author | Yes

# ToDo

Community contributions are welcome.

* Several filters have extra parameters that could be exposed (drive, thermal tuning coefficients, etc).
* Many filters may also be easily modified for HPF or other types of output.

# License
If not otherwise stated in the header of a file, all other code in this project is released under the Unlicense.
