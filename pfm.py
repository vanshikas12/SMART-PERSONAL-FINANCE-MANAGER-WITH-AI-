import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import messagebox

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Expense Tracker Class
class ExpenseTracker:
    def __init__(self, csv_file='C:\\Users\\kinsh\\Downloads\\VerveBridge-Smart_Personal_Finance_Manager-main\\VerveBridge-Smart_Personal_Finance_Manager-main\\expenses.csv'):
        self.csv_file = csv_file
        try:
            self.expenses = pd.read_csv(csv_file, parse_dates=['Date'])
        except FileNotFoundError:
            self.expenses = pd.DataFrame(columns=['Date', 'Category', 'Description', 'Amount'])

    def add_expense(self, date, category, description, amount):
        new_expense = pd.DataFrame({'Date': [date], 'Category': [category], 'Description': [description], 'Amount': [amount]})
        self.expenses = pd.concat([self.expenses, new_expense])
        self.save_to_csv()

    def view_expenses(self):
        return self.expenses

    def calculate_totals(self):
        self.expenses['Date'] = pd.to_datetime(self.expenses['Date'])
        daily_totals = self.expenses.resample('D', on='Date')['Amount'].sum()
        monthly_totals = self.expenses.resample('M', on='Date')['Amount'].sum()
        yearly_totals = self.expenses.resample('Y', on='Date')['Amount'].sum()
        return daily_totals, monthly_totals, yearly_totals

    def visualize_expenses_by_category(self):
        if self.expenses.empty:
            print("No expenses to display.")
            return
        
        category_totals = self.expenses.groupby('Category')['Amount'].sum()
        category_totals.plot(kind='bar', color='skyblue')
        plt.title('Expenses by Category')
        plt.xlabel('Category')
        plt.ylabel('Total Amount')
        plt.show()

    def visualize_expenses_over_time(self):
        if self.expenses.empty:
            print("No expenses to display.")
            return
        
        self.expenses['Date'] = pd.to_datetime(self.expenses['Date'])
        self.expenses.set_index('Date', inplace=True)
        daily_totals = self.expenses.resample('D')['Amount'].sum()
        daily_totals.plot(kind='line', marker='o', color='purple')
        plt.title('Expenses Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Amount')
        plt.show()

    def save_to_csv(self):
        self.expenses.to_csv(self.csv_file, index=False)

# Budget Management Class
class BudgetManager:
    def __init__(self):
        self.budget = 0

    def set_budget(self, budget):
        self.budget = budget

    def check_budget(self, amount):
        if amount > self.budget:
            return False
        else:
            return True

# Financial Goal Setting Class
class FinancialGoal:
    def __init__(self):
        self.goals = []

    def add_goal(self, goal):
        self.goals.append(goal)

    def view_goals(self):
        return self.goals

# AIAssistant Class with OpenAI GPT Integration
class AIAssistant:
    def __init__(self, api_key):
        openai.api_key = api_key

    def answer_query(self, query):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant."},
                    {"role": "user", "content": query}
                ]
            )
            answer = response['choices'][0]['message']['content'].strip()
            return answer
        except openai.error.RateLimitError:
            return "Error: API quota exceeded. Please try again later."
        except openai.error.OpenAIError as e:
            return f"Error: {str(e)}"

# Simple NLP-Based Chatbot
class SimpleChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.responses = {
            'hello': "Hi! How can I assist you with your finances today?",
            'budget': "To set or check your budget, please use the relevant buttons in the application.",
            'expense': "You can add, view, and visualize your expenses using the provided options.",
            'goal': "To add or view financial goals, please use the corresponding buttons."
        }

    def preprocess_query(self, query):
        tokens = word_tokenize(query.lower())
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def answer_query(self, query):
        tokens = self.preprocess_query(query)
        response = "I'm not sure how to respond to that. Could you please clarify?"
        for token in tokens:
            if token in self.responses:
                response = self.responses[token]
                break
        return response

# GUI Class with CSV Support, Visualization, OpenAI Integration, and Simple NLP Chatbot
class GUI:
    def __init__(self, master, api_key):
        self.master = master
        self.master.title("Smart Personal Finance Management Application")
        self.expense_tracker = ExpenseTracker()
        self.budget_manager = BudgetManager()
        self.financial_goal = FinancialGoal()
        self.ai_assistant = AIAssistant(api_key)
        self.simple_chatbot = SimpleChatbot()

        # Create GUI components
        self.date_label = tk.Label(master, text="Date")
        self.date_label.pack()
        self.date_entry = tk.Entry(master)
        self.date_entry.pack()

        self.category_label = tk.Label(master, text="Category")
        self.category_label.pack()
        self.category_entry = tk.Entry(master)
        self.category_entry.pack()

        self.description_label = tk.Label(master, text="Description")
        self.description_label.pack()
        self.description_entry = tk.Entry(master)
        self.description_entry.pack()

        self.amount_label = tk.Label(master, text="Amount")
        self.amount_label.pack()
        self.amount_entry = tk.Entry(master)
        self.amount_entry.pack()

        self.add_expense_button = tk.Button(master, text="Add Expense", command=self.add_expense)
        self.add_expense_button.pack()

        self.view_expenses_button = tk.Button(master, text="View Expenses", command=self.view_expenses)
        self.view_expenses_button.pack()

        self.visualize_category_button = tk.Button(master, text="Visualize by Category", command=self.visualize_expenses_by_category)
        self.visualize_category_button.pack()

        self.visualize_time_button = tk.Button(master, text="Visualize Over Time", command=self.visualize_expenses_over_time)
        self.visualize_time_button.pack()

        self.set_budget_button = tk.Button(master, text="Set Budget", command=self.set_budget)
        self.set_budget_button.pack()

        self.check_budget_button = tk.Button(master, text="Check Budget", command=self.check_budget)
        self.check_budget_button.pack()

        self.add_goal_button = tk.Button(master, text="Add Goal", command=self.add_goal)
        self.add_goal_button.pack()

        self.view_goals_button = tk.Button(master, text="View Goals", command=self.view_goals)
        self.view_goals_button.pack()

        self.query_label = tk.Label(master, text="Query")
        self.query_label.pack()
        self.query_entry = tk.Entry(master)
        self.query_entry.pack()

        self.answer_button = tk.Button(master, text="Get Answer", command=self.get_answer)
        self.answer_button.pack()

        self.chatbot_query_label = tk.Label(master, text="Chatbot Query")
        self.chatbot_query_label.pack()
        self.chatbot_query_entry = tk.Entry(master)
        self.chatbot_query_entry.pack()

        self.chatbot_answer_button = tk.Button(master, text="Chatbot Answer", command=self.get_chatbot_answer)
        self.chatbot_answer_button.pack()

    def add_expense(self):
        date = self.date_entry.get()
        category = self.category_entry.get()
        description = self.description_entry.get()
        amount = float(self.amount_entry.get())
        self.expense_tracker.add_expense(date, category, description, amount)
        messagebox.showinfo("Success", "Expense added successfully!")

    def view_expenses(self):
        expenses = self.expense_tracker.view_expenses()
        messagebox.showinfo("Expenses", str(expenses))

    def visualize_expenses_by_category(self):
        self.expense_tracker.visualize_expenses_by_category()

    def visualize_expenses_over_time(self):
        self.expense_tracker.visualize_expenses_over_time()

    def set_budget(self):
        budget = float(self.amount_entry.get())
        self.budget_manager.set_budget(budget)
        messagebox.showinfo("Success", "Budget set successfully!")

    def check_budget(self):
        amount = float(self.amount_entry.get())
        if self.budget_manager.check_budget(amount):
            messagebox.showinfo("Success", "You are within budget!")
        else:
            messagebox.showerror("Error", "You are over budget!")

    def add_goal(self):
        goal = self.description_entry.get()
        self.financial_goal.add_goal(goal)
        messagebox.showinfo("Success", "Goal added successfully!")

    def view_goals(self):
        goals = self.financial_goal.view_goals()
        messagebox.showinfo("Goals", str(goals))

    def get_answer(self):
        query = self.query_entry.get()
        answer = self.ai_assistant.answer_query(query)
        messagebox.showinfo("Answer", str(answer))

    def get_chatbot_answer(self):
        query = self.chatbot_query_entry.get()
        answer = self.simple_chatbot.answer_query(query)
        messagebox.showinfo("Chatbot Answer", str(answer))

# Running the GUI with API Key
api_key = "your_openai_api_key_here"
root = tk.Tk()
gui = GUI(root, api_key)
root.mainloop()
