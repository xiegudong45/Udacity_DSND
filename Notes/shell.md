#Shell Workshop
## Commands
### Print text
* **`echo`** : print messages in terminal

<center>echo 'Hello!!'</center>

### Print contents in directory
* **`ls`** : print contents in the current directory
* **`ls .`** : print contents in the current directory
* * **`ls ~`** : print contents in home directory
* **`ls dir_name`** : print contents in dir_name
* **`cd dir_name`** : move to dir_name
* **`cd dir_name; ls`** : move to dir_name and print contents in the current directory.
* **`ls dir_name/../dir_name`** : print contents in dir_name.
* **`ls -l dir_name/*.pdf`** : print full information of all file names with .pdf at the end in the dir_name

### Print working directory
* **`pwd`**

### Organizing files
* **`mkdir dir_name`** : make directory
* **`mv curr_dir/file target_dir`** : move files

**When using quotes to represent a directory, star will only be a star.**

### Downloading files
* **`curl -L 'url'`** : follow redirects
* **`curl -o file_name -L 'url'`** : download the content into file_name

### Viewing files
* **`cat file_name`** : show the contents in the file_name
* **`less file_name`** : show one screen of the contents in the file_name (use `space` or `->` to show the following content. `q` to quit)

### Removing things
* **`rm file_name`** : remove file_name (cannot be taken back)
* **`rm -i file_name`** : need confirmation before remove file_name (cannot be taken back)
* **`rmdir dir_name`**: remove dir_name (cannot be taken back)

### Searching and pipes
* **`grep word file_name`** : print all the lines which contains word
* **`grep word file_name | less`** : print the lines which contains word in one page.
* **`grep word file_name | wc -l`** : count word appeared in the file
* **`grep -c word file_name`** : count word appeared in the file

### Shell and environment variables
* **assign an variable** : var_name='abc'
* **print the variable value** : echo $var_name
* **Shell variable** : Internal to the shell program
* **environment variable** : tells your system where your program files are.
* **add a dir_name to your path** : `PATH=$PATH:/new/dir`

### Startup files(.bash_profile)

###Controlling the shell prompt
* http://bashrcgenerator.com/
* change .bash_profile in Mac or Windows or .bashrc in Linux :
`nano .bash_profile(.bashrc)`
* add `export PS1 ...`

### Aliases
* **`alias ll='ls -la'`**
* **`aliases`** list all aliases
