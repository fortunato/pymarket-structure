import { ChangeDetectionStrategy, Component, ElementRef, input, signal, viewChild } from '@angular/core';

@Component({
	selector: 'app-hint',
	template: `
		<span class="hint" #icon (mouseenter)="show($event)" (mouseleave)="hide()">?</span>
		@if (visible()) {
			<div class="tip" [style.top.px]="tipTop()" [style.right.px]="tipRight()">{{ text() }}</div>
		}
	`,
	styleUrl: './hint.component.scss',
	changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HintComponent {
	readonly text = input.required<string>();
	readonly visible = signal(false);
	readonly tipTop = signal(0);
	readonly tipRight = signal(0);
	private readonly icon = viewChild<ElementRef<HTMLElement>>('icon');

	show(_event: MouseEvent): void {
		const el = this.icon()?.nativeElement;
		if (!el) return;
		const rect = el.getBoundingClientRect();
		this.tipTop.set(rect.bottom + 6);
		this.tipRight.set(window.innerWidth - rect.right);
		this.visible.set(true);
	}

	hide(): void {
		this.visible.set(false);
	}
}
