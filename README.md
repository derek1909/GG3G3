# GG3_G5

How to Collaborate Using Git:

## 1. Clone the Remote Repository

Clone this remote repository to your local folder. `<directory>` is optional, indicating the local folder that you want to clone to. 

```bash
git clone https://github.com/derek1909/GG3G3.git <directory>
```

## 2. Create a New Branch

When you want to work on the project (e.g. task 1.1), create a new branch so that your changes won't affect the main branch before `rebase`.

```bash
git checkout -b <my-branch>
```

Or 

```bash
git switch -C <my-branch>
```

## 3. Make Changes and Commit

First, check the difference between what git has stored and what's on your disk. These are the changes you have made.

```bash
git diff
```

Then, save your changes to your local git and in your new branch.

```bash
git add <changed_files>
git commit -m "what you have done"
```

## 4. Sync Your Local Commit to the Remote Repository

This will add a new branch `<my-branch>` in the remote repository and push your local commits.

```bash
git push origin <my-branch>
```

## 5. Rebase

When you develop a feature in a separate branch, other people might update the main branch with their changes (commits). This means your feature branch might be based on an outdated version of the main branch. Rebase helps you solve this problem. It moves or "rebases" your entire feature branch to begin on the tip of the main branch, effectively incorporating all the new commits from the main branch.

### Fetch the Latest Changes

First, you need to fetch the latest changes of the main branch. In your local repository, you can switch to the main branch and use the `pull` command to fetch the latest changes.

```bash
git checkout main
git pull origin main
```

### Start the Rebase

Then, you can switch back to your working branch and use the `rebase` command to base your changes on top of the latest changes of the main branch.

```bash
git checkout <my-branch>
git rebase main
```

### Resolve Conflicts and Continue Rebase

If conflicts occur during the `rebase`, Git will pause the `rebase` and allow you to resolve the conflicts. You can open the conflicting files in an editor, manually resolve the conflicts, then add the resolved files to the staging area.

```bash
git add filename
git rebase --continue
```

If you want to abandon the `rebase`, you can use the `git rebase --abort` command.

### Push Changes

Once the `rebase` is complete, you can push your changes to the remote repository. Since `rebase` changes the commit history, you need to use the `-f` or `--force` option to force the push.

```bash
git push origin <my-branch> -f
```

After this, you can create a **pull request** on GitHub to request that your branch be merged into the main branch (I will do the merge as soon as possible). Remember, using `rebase` can make the commit history cleaner, but it also makes the history of the changes more linear, which might not always be desirable depending on the project's needs.

## Other Useful Git Commands

(To be continued...)
