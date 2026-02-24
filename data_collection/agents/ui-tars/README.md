# UI-TARS Agent Setup Guide

Prerequisites: You need to have Node.js and npm installed on your machine. You also need an VolcEngine api key.

1. `cd` to an empty directory (not necessarily this one and better a separate one), and:
```bash
npm init -y
npm install @ui-tars/cli uuid
```

2. Apply all bugfix patches under patches/. Since the patches' paths are relative, you need to create a git repository from the directory where you ran `npm install` and then apply the patches. For example:
```bash
patch -p1 < patches/*.patch
```


3. write in your `~/.ui-tars-cli.json`(yes, in the home directory):
```json
{
  "baseURL": "https://ark.cn-beijing.volces.com/api/v3",
  "apiKey":  "your api key CHANGE THIS",
  "model":   "doubao-1-5-ui-tars-250428",
  "useResponsesApi": false
}
```

Note: `doubao-1-5-ui-tars-250428` is going to be deprecated. For replacement please refer to [link](https://www.volcengine.com/docs/82379/1350667?lang=en#:~:text=doubao%2D1%2D5%2Dui,seed%2D1%2D8%2D251228).

3. Launch in the directory path:
```bash
npx @ui-tars/cli start -t adb -q "Help me add Tom to my contacts. His phone number is 12345678900."
```

4. Set up a conda environment from requirements.txt (what our fake adb needs). Then write the directory path and configuration to data_collection/automations_agents.json.

## Appendix

https://github.com/bytedance/UI-TARS-desktop/tree/main/packages/ui-tars/cli

Here’s the deal: the `-p` flag is not for passing raw JSON, it’s for pointing the CLI at a **URL** that returns a YAML "preset" (with `vlmBaseUrl`, `vlmApiKey`, `vlmModelName`, etc). When you passed your JSON string directly it tried to `fetch('{"baseURL":…}')` and blew up with "Only absolute URLs are supported."

You have two easy paths:

1. Persist your config locally once, then drop `-p` entirely  
   ```shell
   # write your config once to ~/.ui-tars-cli.json
   cat > ~/.ui-tars-cli.json <<-EOF
   {
     "baseURL": "https://ark.cn-beijing.volces.com/api/v3",
     "apiKey":   "your api key CHANGE THIS",
     "model":    "doubao-1-5-ui-tars-250428",
     "useResponsesApi": false
   }
   EOF

   # now just invoke without -p
   npx @ui-tars/cli start \
     -t adb \
     -q "Help me add Tom to my contacts. His phone number is 12345678900."
   ```

2. Host a small YAML preset somewhere (e.g. GitHub raw) and point `-p` at it  
   ```yaml
   # presets.yaml (host this on GitHub, your CDN, etc)
   vlmBaseUrl:       https://ark.cn-beijing.volces.com/api/v3
   vlmApiKey:        your api key CHANGE THIS
   vlmModelName:     doubao-1-5-ui-tars-250428
   useResponsesApi:  false
   ```
   ```shell
   npx @ui-tars/cli start \
     -p https://raw.githubusercontent.com/you/your-repo/main/presets.yaml \
     -t adb \
     -q "Help me add Tom to my contacts. His phone number is 12345678900."
   ```

Either way, **`-p` must be a valid HTTP(S) URL** (pointing at a YAML), or you skip it and let the CLI read `~/.ui-tars-cli.json`.