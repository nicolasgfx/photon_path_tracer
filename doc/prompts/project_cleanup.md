- do a FULL project cleanup and code revision:
1. project files & folders (exclude /scenes, /output/, /doc )
2. remove unused testx and test files
3. remove cached and temporary  files
4. check each code file for unused functions and dead code 
5. simplify the code as much as possible. merge files into one, if there is shared context. keep code human readable. respect c++ coding standards. avoid template useage. 
6. I created a back, so we can safely restore files if needed
7. run tests to see if you have regression
