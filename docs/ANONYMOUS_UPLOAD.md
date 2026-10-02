# Publishing the review snapshot

1. Create a new GitHub repository with a neutral name such as `review-artifact`
   and upload the contents of this directory.
   Use a fresh repository: do not import the original repositories or their
   history. This snapshot contains no original Git metadata.
2. Open https://anonymous.4open.science/anonymize and provide that repository URL.
   Configure the repository owner, account names, affiliations, and other
   identifying terms to be hidden. `MOBench` is the method name and Python
   import namespace; do not add it to the identifying terms. Enable download
   access when creating the
   anonymous mirror.
3. Open the generated review URL in a signed-out browser. Check the README,
   source files, downloads, and relative links. Only the anonymous URL should
   appear in the submitted paper.
4. Freeze the reviewed source revision. If you update this snapshot, refresh the
   mirror and verify its content again before giving reviewers the link.

The service mirrors a GitHub repository. The ZIP supplied with this snapshot is
for transporting the files; unpack it before uploading its contents to GitHub.

Service documentation: https://github.com/tdurieux/anonymous_github
Conference instructions: https://conf.researchr.org/track/fse-2027/fse-2027-papers
