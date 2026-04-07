#pragma once
struct pllStack {
  void *item;
  struct pllStack *next; // node ben duoi trong stack
};

void pllStackClear(pllStack **stack);
void *pllStackPop(pllStack **head);
int pllStackPush(pllStack **head, void *item);