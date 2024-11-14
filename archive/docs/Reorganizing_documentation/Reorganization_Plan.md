# Documentation reorganization plan

### Goals for Programmer documentation

1. **completeness** Documentation reflects code: each .py and each class is referenced and explained in the documentation. 
2. **Readability** All terms need to be explained before they are used, examples: django, controller, URL ...
3. **limited scope** Each document file should explain a limited, well define subject. Different subjects should be described in different files.
4. **information hiding** There is a consistent hierarchy in the documentation where each level hides non-critical aspects of the code below it, exposes minimal set of parameters that are used at higher levels. 
5. **Automatic generation**, using tools such as Sphinx the documentation is combined with the code. A compiler can then be used that translates the documentation into a easy to navvigate web format.
Ed has done this using sphinx and readthedocs: https://activebrainatlasadmin.readthedocs.io/en/latest/index.html

## Plan
We start with Ed's documentation and add to it the content for the currently undocumented packages.

## Issues with current programmer documentation.

1. **Incompleteness:** The file lsr-code.txt lists all of the files in the pipeliine directory, there are about 1000 files in this directory. I found no reference to any individual files in the programmer documentation.
2. **confusing terminology** Are we calling *all* of the code the "pipeline", or is the piepline specific to what is sometime called the "czi.2.neuroglancer" code? I think that structure detection, marked cell detection and alignment should be in separate directories.
4. The file ` programmer/high.level.overall.design.of.projects.md `  is incoherent and too long. I wrote what I believe is a better version in [high.level.doc.dir/high.level.overall.design.of.projects.md](high.level.doc.dir/high.level.overall.design.of.projects.md)

### Goals for user Documentation

1. **Task Specific** a typical user wants to use the system in a particular way. It is therefor desirable to create a specialized documentation for each workflow. Parts of the workflows can be common, so they can be put in files that are poointed to by the workflow documentation.
2. **Completeness:** The documentation should note assume prior knowledge of the user. The explanation should be step-by-step.
3. **Readability** The explanation should be written in good english and explain all terms before they are used. User documentation quality is judged by the target user.



# Testing Plan

There are two types of testing:

* Internal unit-tests: Tests suits are run through scripts to check each class whether its publc methods work as expected. The tests are written by programmers but are run automatically after each addition or change.
* External Usability tests: Test are done by users (not programmers) to evaluate whether the system as expected.


Currently, we have no systematic unit tests and the extarnal tests are not done systematically.

To organize the internal tests there are well established open-source Frameworks, such as `systest` and `unittest`
