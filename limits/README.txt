LIMITS FOLDER
=============

This folder controls the assistant's behaviour. All files are plain text
and can be edited in any text editor.

Files
-----
  persona.txt   Describes who the assistant is and its general character.
                This is prepended to every conversation.

  allowed.txt   A list of things the assistant should always be willing to do.
                Add lines to expand its willingness to help in specific areas.

  denied.txt    A list of hard limits — things it must never do.
                Add lines here to restrict specific topics or behaviours.

  README.txt    This file.

How it works
------------
When you run `python3 generate.py`, the contents of persona.txt, allowed.txt,
and denied.txt are combined into a system prompt that is silently prepended
to every conversation. The model uses this context to shape its responses.

Editing
-------
Open any file in a text editor and add or remove lines.
Lines starting with # in allowed.txt and denied.txt are treated as comments
and are ignored.
Changes take effect the next time you start generate.py.
