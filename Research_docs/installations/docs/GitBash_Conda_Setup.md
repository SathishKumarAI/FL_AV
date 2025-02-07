[How to add Conda to Git Bash (Windows)*Medium Link*](https://fmorenovr.medium.com/how-to-add-conda-to-git-bash-windows-21f5e5987f3d)

[second_link_references](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473)

new path for anaconda installation is "C:\ProgramData\anaconda3\etc\profile.d" in windows.
-----

**Summary:**

The following commands are designed to enhance your Git Bash environment by appending a line to your `~/.bashrc` file. This line sources the `conda.sh` script, enabling Conda functionality within Git Bash.

**Without Spaces in Path:**
If the path to your current directory (`${PWD}`) lacks spaces, use the following command:

```bash
echo ". ${PWD}/conda.sh" >> ~/.bashrc
```

This command appends a line to `~/.bashrc`, instructing Git Bash to source `conda.sh` on startup.

**With Spaces in Path:**
In case the path contains spaces, enclose the entire path in single quotes:

```bash
echo ". '${PWD}'/conda.sh" >> ~/.bashrc
```

This ensures that the path is treated as a single argument, even with spaces.

**Applying Changes:**
After executing either command, close and reopen Git Bash to activate the changes. Modifications to `~/.bashrc` usually take effect when the shell starts.

**Activate Base Environment by Default:**
To set the base Conda environment as the default in Git Bash:

1. Open Git Bash.

2. Use a text editor like `nano` to open `~/.bashrc`:

    ```bash
    nano ~/.bashrc
    ```

3. Navigate to the end of the file and add:

    ```bash
    conda activate base
    ```

4. Save and exit the editor (`Ctrl` + `X`, `Y`, `Enter` in `nano`).

5. Restart Git Bash:

    ```bash
    exec bash
    ```

Now, the base Conda environment activates automatically on Git Bash startup.

**Note:**
Ensure Conda binaries are in your system's PATH. If issues arise, verify a correct Conda installation and include the installation directory in your PATH.

---