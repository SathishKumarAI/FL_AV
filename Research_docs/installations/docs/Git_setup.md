# Configuring Git User Information in Git Bash

To configure your Git user information in Git Bash, run the following commands:

1. **Set Global User Name:**

    ```bash
    git config --global user.name "devil"
    ```

    This command sets your global Git user name to "devil."

2. **Set Global User Email:**

    ```bash
    git config --global user.email "devil.paloju99@gmail.com"
    ```

    This command sets your global Git user email to "devil.paloju99@gmail.com."

## Additional Information:

- **Purpose:**
  These commands set global configurations for your Git user name and email. It helps identify your commits with the correct authorship information.

- **Global vs. Local Configuration:**
  Using the `--global` flag makes the configuration global, applying it to all your Git repositories. If you want to set user information for a specific repository only, omit the `--global` flag.

- **Check Configurations:**
  To check your Git configurations, you can use the following commands:

  ```bash
  git config --global --get user.name
  git config --global --get user.email
