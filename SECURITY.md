# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities by email to **fodorad201@gmail.com** with the subject line
`[OmniLoader] Security vulnerability`.

Include:
- A description of the vulnerability and its potential impact.
- Steps to reproduce or a minimal proof-of-concept.
- Your suggested fix, if you have one.

You can expect an acknowledgement within **72 hours** and a status update within **7 days**.
If a fix is warranted, it will be released as a patch version and credited to you
(unless you prefer anonymity).

## Scope note

OmniLoader is a **library** that loads data you point it at. It does not download
or execute remote code. When reading HDF5 files, only open files from trusted
sources — as with any deserialization, malformed files could be crafted to
trigger excessive memory use in the underlying `h5py`/HDF5 stack.
