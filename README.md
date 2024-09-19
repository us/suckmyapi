# SuckMyAPI: Max Out Your API Limits (For Fun and Chaos!)

Ever wish you could see just how far an API can go before it throws in the towel? Or maybe you're tired of clients who conveniently forget to pay up. Say hello to **SuckMyAPI**—the most fun (and responsible) way to stress-test APIs like **OpenAI**, **Anthropic**, and more. Push them to their limits and watch the madness unfold, but be prepared for your wallet to scream for mercy!

---

## 🎭 Welcome to API Chaos!

Curious what happens when an API hits its breaking point? **SuckMyAPI** is here to let you find out in the most chaotic way possible. Whether it's sending massive token streams to **ChatGPT** or making **Claude** sweat, SuckMyAPI will show you how far these APIs can be pushed—right up to the edge of their limits.

---

## 🤔 What’s SuckMyAPI?

SuckMyAPI is a devilishly fun tool that pushes top APIs like **OpenAI’s GPT-4** and **Anthropic’s Claude** to the max. With tokens flying, CPUs overheating, and rate limits hit like never before, SuckMyAPI is the ultimate playground for **LLM**, **API**, and **rate limit** explorers. The question is: **How far can you push?**

---

## 🔥 APIs You Can Wreak Havoc On

Currently, we support the following powerhouses:

1. **OpenAI** – See how many tokens you can send before ChatGPT waves the white flag.
2. **Anthropic** – Push Claude to its limit and see how it handles the overload.

### Want More?  
We’re always open to adding more APIs! Got a favorite API that needs testing? Submit a pull request, and let’s turn it into a stress-testing party.

---



## 🎢 Features to Break the Bank (And API Limits)

Here’s where the real fun begins. **SuckMyAPI** takes the concept of API usage and cranks it up to 11. This isn't just about making a few requests here and there—no, we’re talking about **pushing every single API endpoint to the max** and **sending requests asynchronously** at breakneck speed. Here’s how we make it happen:

### 1. **Token Tsunami**
Ever wonder what happens when you flood an API with tokens non-stop? With **SuckMyAPI**, you’ll find out! We leverage each API’s token handling capacity to the extreme. For example, if **OpenAI’s GPT-4** has a max token limit per request, we hit it—again and again, until the API cries uncle. The same goes for **Anthropic’s Claude**. Our token tsunami overloads each model with input and output tokens, making sure every endpoint gets a workout.

### 2. **Limit Limbo**
API rate limits? Pfft. Let’s see how far those limits can bend. Every API comes with a set number of requests allowed per minute, hour, or day. Whether it’s **OpenAI**, **Anthropic**, or any other API you implement, SuckMyAPI fires requests asynchronously, maxing out those limits across all endpoints. The best part? All requests are sent simultaneously, meaning we’re hitting those rate limits like a ton of bricks—**async-style**, no holding back.

- **OpenAI’s GPT-4**: Test the per-minute request limit across multiple endpoints—**completions**, **edits**, and **embeddings**—all at once.
- **Anthropic’s Claude**: Push through the token limit while also testing rate limits across Claude’s various tasks.

### 3. **Bill Buster 3000**
This is where it gets painful (and hilarious). Every API has a billing model based on usage—whether it’s tokens, requests, or time. Our **Bill Buster 3000** feature pushes these models to the edge, showing you exactly how much it’ll cost to run full-throttle requests. Want to see your API budget go up in flames? SuckMyAPI makes it happen. The algorithm doesn’t just maximize tokens or requests—it calculates the cost of each batch in real-time, so you can watch the numbers rack up as you push the limits.

### 4. **Endpoint Explorer**
We don’t just test a single endpoint—we test **ALL of them**. With **SuckMyAPI**, every available API endpoint gets the same maxed-out treatment, ensuring that you’re stress-testing your entire API integration, not just a single task. From completions and edits in **OpenAI**, to summarization or dialogue generation in **Claude**, every API call is pushed to the breaking point, in parallel, at maximum capacity.

---

**How It Works:**

- **Async Requests**: SuckMyAPI uses asynchronous programming to make API requests. This means we can send a barrage of calls to every endpoint—**simultaneously**—maxing out API limits faster than you can blink.
  
- **Maxing out Per-Minute Limits**: Each API typically allows a set number of requests per minute. SuckMyAPI tests those limits by firing as many requests as possible in the shortest amount of time.
  
- **Endpoint Variety**: Why stop at one endpoint? We hit them all. Want to overload **OpenAI’s Completions** while simultaneously flooding **Embeddings** and **Moderations**? Done. Want to hit **Claude’s dialogue generation** and summarization at the same time? Easy.

The result? A **full-throttle stress test** that shows you exactly how far your favorite APIs can go before they break—or you run out of credits!

---

## ⚠️ Important Warnings

1. **💸 Bill Shock Alert**: Yes, it’s fun, but expect some **serious billing damage**. Prepare your wallet for impact.
2. **🔥 Meltdown Incoming**: If your servers (or the API provider’s) start smoking, well, don’t say we didn’t warn you.
3. **👮 Use Responsibly**: With great power comes great *"Oh no, what have I done?"* moments. Please don’t overdo it.

---

## 🛠️ Setting Up SuckMyAPI

Ready to break some APIs? Here’s how to set things up:

1. **Clone the repo**:
   ```
   git clone https://github.com/your-username/suckmyapi.git
   cd suckmyapi
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Configure your API keys**:
   In your `.env` file, add your API keys:
   ```  
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

   You can also implement any additional API you want to wreck—just be sure to adjust the code and configuration!

---

## 🚀 How to Run the Madness

Once you’re ready to dive into the chaos, simply run:

```
python suck.py
```

You’ll be asked to choose between OpenAI and Anthropic:

```
Choose an option:
1. OpenAI
2. Anthropic
3. Both 😈
Enter the number of your choice: 
```

Pick your poison—**1** for OpenAI, **2** for Anthropic, or **3** for Both—and let the stress-testing begin!

---

## 📊 Sample Output (Brace Yourself)

Here’s an idea of what you might see after a typical **SuckMyAPI** session:
```
Batch Summary:
--------------
claude-3-opus: sent 12, ok/fail 12/0, responded 12/12
claude-3-sonnet: sent 12, ok/fail 12/0, responded 12/12
Total input tokens: 1621  
Total output tokens: 31,757  
Estimated total cost: $183.4568  
```
That’s right—**$183** for a single batch! Who needs a savings account when you’ve got SuckMyAPI?

---

## 💡 Pro Tip: Open the API Floodgates

Want to stress-test a different API? Fork the repo, implement your API, and get ready to watch their servers panic when you unleash a wave of requests.

---

## 🧪 Why Use SuckMyAPI? (For Science, Of Course)

Here’s why SuckMyAPI is the ultimate experiment tool:
- **API Resilience Testing**: Just how much abuse can that API handle before it taps out?
- **Simulate Heavy Traffic**: Ever dreamed of your startup getting the same load as Google? Now’s your chance to pretend.
- **Test “Unlimited” API Plans**: Let’s see just how “unlimited” those API plans really are.

**Important Note**: While this is all in good fun, it can get expensive, so maybe keep your accountant in the loop!

---

# 🤝 Get Involved: Join the API Chaos Crew!

Got a killer idea to make SuckMyAPI even better? Maybe another API to test? We’re all about chaos, and we’d love your help. Submit a pull request or open an issue, and let’s turn up the API madness together!

### Why Contribute?
- Gain feedback to make your project even better.
- See your ideas implemented in an insane project.
- Earn API chaos fame and recognition!

---

## 📜 License

This project is licensed under [GLWTPL](https://github.com/us/hark/blob/master/LICENSE) (GOOD LUCK WITH THAT PUBLIC LICENSE).

---

## ⚠️ Disclaimer

SuckMyAPI is for **educational purposes** only. If you choose to use it for other reasons (you little troublemaker), we’re not responsible for the consequences. Please don’t use it for evil purposes!

---

# 📬 Reach Out to Us!

Got questions, suggestions, or just want to chat about API madness? Hit us up at [rahmetsaritekin@gmail.com](mailto:rahmetsaritekin@gmail.com).

**Now go forth, have fun, and break some APIs! (Just don’t tell them we sent you.)**

---

This content is optimized for:
- **API stress testing**
- **OpenAI API usage**
- **Anthropic Claude testing**
- **ChatGPT overload**
- **Token limits**
- **Rate limit testing**
- **API billing costs**

Make sure you explore how far APIs can go—with **SuckMyAPI**!













