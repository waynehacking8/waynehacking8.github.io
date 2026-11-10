---
title: GB10 Personal Agent Demo, privacy
description: What the private GB10 Personal Agent demo reads from Google, where it runs, and how to revoke access.
---

# GB10 Personal Agent Demo: privacy

This is a private demo I built and run myself on NVIDIA GB10 hardware. It is not a public product, it has no sign-up, and only one Google account ever connects to it: a demo account I control that holds made-up mail and calendar entries.

## What it reads and writes

With the account's consent the demo uses three Google scopes.

* `gmail.readonly`: it searches and reads mail in that account.
* `calendar.freebusy`: it checks when the account is busy.
* `calendar.events.owned`: it creates a calendar event, and only after I press Allow on an approval card in the app.

It sends no mail and deletes nothing.

## Where the data goes

The language model runs on the same machine as the app. Mail and calendar text is not sent to any outside AI service, and nothing is shared, sold, or used for advertising or model training. Mail headers can sit in the machine's memory for up to 12 hours to make repeat searches faster; they are never written to disk. The Google access token is kept as a file on that machine, readable only by the user account the app runs under and the machine's administrators.

## Revoking access

Open [myaccount.google.com/permissions](https://myaccount.google.com/permissions), pick the app named GB10 Personal Agent Demo, and remove access. I also revoke the token myself when the demo is retired.

## Contact

Wayne Chiu, [waynehacking8@gmail.com](mailto:waynehacking8@gmail.com).
