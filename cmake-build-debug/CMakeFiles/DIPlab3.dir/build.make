# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2020.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion-2020.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mrdragon/Code/C/SLIC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mrdragon/Code/C/SLIC/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/DIPlab3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DIPlab3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DIPlab3.dir/flags.make

CMakeFiles/DIPlab3.dir/main.cpp.o: CMakeFiles/DIPlab3.dir/flags.make
CMakeFiles/DIPlab3.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mrdragon/Code/C/SLIC/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DIPlab3.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DIPlab3.dir/main.cpp.o -c /home/mrdragon/Code/C/SLIC/main.cpp

CMakeFiles/DIPlab3.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DIPlab3.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mrdragon/Code/C/SLIC/main.cpp > CMakeFiles/DIPlab3.dir/main.cpp.i

CMakeFiles/DIPlab3.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DIPlab3.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mrdragon/Code/C/SLIC/main.cpp -o CMakeFiles/DIPlab3.dir/main.cpp.s

# Object files for target DIPlab3
DIPlab3_OBJECTS = \
"CMakeFiles/DIPlab3.dir/main.cpp.o"

# External object files for target DIPlab3
DIPlab3_EXTERNAL_OBJECTS =

DIPlab3: CMakeFiles/DIPlab3.dir/main.cpp.o
DIPlab3: CMakeFiles/DIPlab3.dir/build.make
DIPlab3: /usr/local/lib/libopencv_highgui.so.4.5.1
DIPlab3: /usr/local/lib/libopencv_videoio.so.4.5.1
DIPlab3: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
DIPlab3: /usr/local/lib/libopencv_imgproc.so.4.5.1
DIPlab3: /usr/local/lib/libopencv_core.so.4.5.1
DIPlab3: /usr/local/lib/libopencv_cudev.so.4.5.1
DIPlab3: CMakeFiles/DIPlab3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mrdragon/Code/C/SLIC/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable DIPlab3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DIPlab3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DIPlab3.dir/build: DIPlab3

.PHONY : CMakeFiles/DIPlab3.dir/build

CMakeFiles/DIPlab3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DIPlab3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DIPlab3.dir/clean

CMakeFiles/DIPlab3.dir/depend:
	cd /home/mrdragon/Code/C/SLIC/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mrdragon/Code/C/SLIC /home/mrdragon/Code/C/SLIC /home/mrdragon/Code/C/SLIC/cmake-build-debug /home/mrdragon/Code/C/SLIC/cmake-build-debug /home/mrdragon/Code/C/SLIC/cmake-build-debug/CMakeFiles/DIPlab3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DIPlab3.dir/depend

