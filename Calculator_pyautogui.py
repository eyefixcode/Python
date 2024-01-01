import tkinter as tk

def on_click(button_text):
    current_text = entry_var.get()
    
    if button_text == "=":
        try:
            result = eval(current_text)
            entry_var.set(result)
        except Exception as e:
            entry_var.set("Error")
    elif button_text == "C":
        entry_var.set("")
    else:
        entry_var.set(current_text + button_text)

# Create the main window
root = tk.Tk()
root.title("Simple Calculator")

# Entry widget for displaying and entering numbers
entry_var = tk.StringVar()
entry = tk.Entry(root, textvariable=entry_var, justify="right", font=("Helvetica", 16))
entry.grid(row=0, column=0, columnspan=4, sticky="nsew")

# Buttons
buttons = [
    '7', '8', '9', '/',
    '4', '5', '6', '*',
    '1', '2', '3', '-',
    '0', '.', '=', '+'
]

row_val = 1
col_val = 0

for button_text in buttons:
    tk.Button(root, text=button_text, padx=20, pady=20, font=("Helvetica", 14), command=lambda btn=button_text: on_click(btn)).grid(row=row_val, column=col_val)
    col_val += 1
    if col_val > 3:
        col_val = 0
        row_val += 1

# Configure row and column weights so that they expand proportionally
for i in range(5):
    root.grid_rowconfigure(i, weight=1)
    root.grid_columnconfigure(i, weight=1)

root.mainloop()
