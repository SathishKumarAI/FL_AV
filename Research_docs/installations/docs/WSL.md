

* ### Enable WSL Feature:

*  ### Enable Virtual Machine Platform (Optional):

Restart your computer to apply the changes.

Open the Microsoft Store.
Search for your preferred Linux distribution (e.g., Ubuntu, Debian, Fedora) and install it.

Launch the installed Linux distribution from the Start menu.

If you want to use WSL 2, you need to set it as your default version.


```wsl --set-default-version 2```

Download and Install WSL 2 Linux Kernel Update Package:

* ### Restart Your Computer

Given username: siuadmin
password : devil@123
password: siuadmin


- [Link for an error, i faced](https://answers.microsoft.com/en-us/windows/forum/all/fullyqualifiederrorid-unauthorizedaccess/a73a564a-9870-42c7-bd5e-7072eb1a3136)


Creating Root User in Ubuntu 22.04.3 LTS

```sudo adduser siuadmin```

```sudo usermod -aG sudo siuadmin```

```su - siuadmin```

```sudo ls /root```

```sudo -i```